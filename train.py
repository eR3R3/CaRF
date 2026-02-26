import os
import torch
import matplotlib.pyplot as plt
from random import randint
from utils.loss_utils import l1_loss, ssim,bce_loss,dice_loss,multi_pos_cross_entropy
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
import torch.nn.functional as F
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
    

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from,epoch, original_args=None):
    
    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    if opt.include_feature:
        if not checkpoint:
            raise ValueError("checkpoint missing!!!!!")
    if checkpoint:
        checkpoint_data = torch.load(checkpoint)
        if isinstance(checkpoint_data, dict):
            model_params = checkpoint_data.get('model_params', checkpoint_data)
            first_iter = checkpoint_data.get('iteration', 0)
        else:
            (model_params, first_iter) = checkpoint_data
        if len(model_params) == 12 and opt.include_feature:
            first_iter = 0
        gaussians.restore(model_params, opt)
        
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color,dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    
    train_cams = scene.getTrainCameras().copy()
    steps_per_epoch = sum(len(cam.sentence) for cam in train_cams)

    max_iters = 26000 
    progress_bar = tqdm(total=max_iters, desc="Training progress")

    first_iter += 1
    iteration = 0
    ratio = 0.1
    total_loss = []
    for epoch in range(epoch_num):
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        while len(viewpoint_stack)!=0:
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
            text_feature=gaussians.get_text(viewpoint_cam.sentence).to("cuda")
            
            for i in range(len(viewpoint_cam.sentence)):
                iter_start.record()
                render_pkg = render(viewpoint_cam, gaussians, pipe, background, opt,sentence=viewpoint_cam.sentence[i],ratio=ratio)
                language_feature,mean_tensor=render_pkg["language_feature_image"],render_pkg["mean_tensor"]
                if opt.include_feature:
                    features=gaussians.mlp1(text_feature)
                    features=torch.mean(features, dim=1)
                    mean_tensor=F.normalize(mean_tensor,dim=1)
                    features=F.normalize(features,dim=1)

                    cosine_similarities=(torch.matmul(mean_tensor,features.T)/0.1).to("cuda")
                    
                    sentence_tensor = torch.zeros(len(viewpoint_cam.sentence))
                    
                    sentence_tensor[i] = 1
                    current_category = viewpoint_cam.category[i]
                    category_indices = [idx for idx, cat in enumerate(viewpoint_cam.category) if cat == current_category]
                    sentence_tensor[category_indices] = 1
                    sentence_tensor = sentence_tensor.unsqueeze(0).to("cuda")
                    com_loss = multi_pos_cross_entropy(cosine_similarities, sentence_tensor)
                    gt_mask = viewpoint_cam.gt_mask[viewpoint_cam.category[i]].to("cuda")
                    
                    # 计算当前视角的bce loss
                    current_bce_loss = bce_loss(language_feature, gt_mask)
                    
                    second_view_bce_loss = 0
                    for other_cam in train_cams:
                        if other_cam != viewpoint_cam and current_category in other_cam.category:
                            # 找到包含相同category的另一个视角
                            other_sentence_idx = other_cam.category.index(current_category)
                            other_sentence = other_cam.sentence[other_sentence_idx]
                            
                            # 对第二个视角进行渲染
                            other_render_pkg = render(other_cam, gaussians, pipe, background, opt, 
                                                    sentence=other_sentence, ratio=ratio)
                            other_language_feature = other_render_pkg["language_feature_image"]
                            other_gt_mask = other_cam.gt_mask[current_category].to("cuda")
                            
                            # 计算第二个视角的bce loss
                            second_view_bce_loss = bce_loss(other_language_feature, other_gt_mask)
                            break 
                    
                    
                    # 合并两个视角的bce loss：当前视角0.7权重，第二个视角0.3权重
                    combined_bce_loss = 0.7 * current_bce_loss + 0.3 * second_view_bce_loss
                    #combined_bce_loss = 1.0 * current_bce_loss
                    loss = combined_bce_loss + 0.1*com_loss
                    loss.backward()
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                iter_end.record()

                # —— 计步、更新进度条与日志 —— #
                iteration += 1
                with torch.no_grad():
                    ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                    progress_bar.update(1)                    # 每步 +1
                    
                    if iteration % 10 == 0:
                        progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.7f}"})
                    total_loss.append(ema_loss_for_log)
                
                # 每50个iteration运行一次render.py
                if (iteration % 500 == 0) or (iteration in range(10000, 12000, 50)):
                    print(f"\nRunning render.py at iteration {iteration}")
                    import subprocess
                    import sys
                    import os
                    
                    # 临时保存当前模型状态
                    temp_checkpoint_path = os.path.join(scene.model_path, f"temp_chkpnt_iter_{iteration}.pth")
                    torch.save((gaussians.capture(opt.include_feature), iteration), temp_checkpoint_path)
                    
                    # 构建render.py的调用命令，传递必要的参数
                    render_cmd = [
                        sys.executable, "/root/autodl-tmp/refer-splat/render.py",
                        "--checkpoint", temp_checkpoint_path,
                        "--include_feature"
                    ]
                    
                    # 添加必要的参数
                    if original_args:
                        if hasattr(original_args, 'source_path'):
                            render_cmd.extend(["--source_path", original_args.source_path])
                        if hasattr(original_args, 'model_path'):
                            render_cmd.extend(["--model_path", original_args.model_path])
                        if hasattr(original_args, 'images'):
                            render_cmd.extend(["--images", original_args.images])
                        if hasattr(original_args, 'white_background') and original_args.white_background:
                            render_cmd.append("--white_background")
                    
                    try:
                        # 运行render.py (不捕获输出，让进度条显示)
                        print(f"Running render.py for iteration {iteration}")
                        result = subprocess.run(render_cmd, check=True)
                        print(f"Render completed successfully for iteration {iteration}")
                        
                        # 运行test_miou.py计算分数
                        print(f"Calculating mIoU for iteration {iteration}")
                        miou_cmd = [sys.executable, "/root/autodl-tmp/refer-splat/test_miou.py"]
                        try:
                            miou_result = subprocess.run(miou_cmd, capture_output=True, text=True, check=True)
                            miou_output = miou_result.stdout.strip()
                            print(f"mIoU Result for iteration {iteration}:")
                            print(miou_output)
                            
                            # 保存mIoU分数到文件
                            miou_log_path = os.path.join(scene.model_path, "miou_scores_attention.txt")
                            with open(miou_log_path, "a", encoding="utf-8") as f:
                                f.write(f"Iteration {iteration}:\n")
                                f.write(f"{miou_output}\n")
                                f.write("-" * 50 + "\n")
                            print(f"mIoU scores saved to: {miou_log_path}")
                            
                        except subprocess.CalledProcessError as miou_e:
                            error_msg = f"mIoU calculation failed: {miou_e.stderr}"
                            print(error_msg)
                            # 同样保存错误信息
                            miou_log_path = os.path.join(scene.model_path, "miou_scores_attention.txt")
                            with open(miou_log_path, "a", encoding="utf-8") as f:
                                f.write(f"Iteration {iteration}:\n")
                                f.write(f"{error_msg}\n")
                                f.write("-" * 50 + "\n")
                        except Exception as miou_e:
                            error_msg = f"mIoU unexpected error: {str(miou_e)}"
                            print(error_msg)
                            # 同样保存错误信息
                            miou_log_path = os.path.join(scene.model_path, "miou_scores_attention.txt")
                            with open(miou_log_path, "a", encoding="utf-8") as f:
                                f.write(f"Iteration {iteration}:\n")
                                f.write(f"{error_msg}\n")
                                f.write("-" * 50 + "\n")
                        
                        # 删除临时checkpoint文件
                        if iteration!= 15100 and os.path.exists(temp_checkpoint_path):
                            os.remove(temp_checkpoint_path)
                            print(f"Temporary checkpoint deleted: {temp_checkpoint_path}")
                            
                    except subprocess.CalledProcessError as e:
                        print(f"Render failed for iteration {iteration}: {e.stderr}")
                        # 删除临时文件
                        if os.path.exists(temp_checkpoint_path):
                            os.remove(temp_checkpoint_path)
                    except Exception as e:
                        print(f"Unexpected error during render: {str(e)}")
                        # 删除临时文件
                        if os.path.exists(temp_checkpoint_path):
                            os.remove(temp_checkpoint_path)

                # 检查退出条件
                if iteration >= max_iters:
                    break

                if iteration%2000==0 and ratio>0.005:
                    ratio=ratio*0.6
                    if ratio<0.005:
                        ratio=0.005

            if iteration >= max_iters:
                break 
        if iteration >= max_iters:
            break

    progress_bar.close()
    
    
if __name__ == "__main__":
    # Set up command line argument parser
    torch.set_default_dtype(torch.float32)
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=55555)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1_000, 1_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1_000, 1_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[1_000, 1_000])
    parser.add_argument("--start_checkpoint", type=str, default = '/root/autodl-tmp/RefSplat_hf/kitchenchkpnt30000.pth')
    #parser.add_argument("--start_checkpoint", type=str, default = '/root/tf-logs/chkpnt0.pth')
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print(args)
    args.model_path = args.model_path
    print("Optimizing " + args.model_path)

    safe_state(args.quiet)
    epoch_num=10
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    # 连续训练5次，每次都从同一个初始checkpoint开始
    num_runs = 3
    
    print(f"\n开始连续训练 {num_runs} 次")
    print("=" * 60)
    
    for run_num in range(1, num_runs + 1):
        print(f"\n开始第 {run_num}/{num_runs} 次训练")
        print("-" * 40)
        
        training(lp.extract(args), op.extract(args), pp.extract(args), 
                args.test_iterations, args.save_iterations, args.checkpoint_iterations, 
                args.start_checkpoint, args.debug_from, epoch_num, original_args=args)
        
        print(f"第 {run_num}/{num_runs} 次训练完成")
    
    print("\n所有 {num_runs} 次训练完成!")