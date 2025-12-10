import sys
sys.path.append(sys.path[0] + '/..')
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

# 假设 RSSamVisionEncoder 已经导入
from mmdet.rsprompter.models import RSSamVisionEncoder

# 设备设置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 预训练模型路径
hf_sam_pretrain_name = "work_dirs/sam_cache/sam_vit_base"
hf_sam_pretrain_ckpt_path = "work_dirs/sam_cache/sam_vit_base/pytorch_model.bin"

# 加载预训练模型
teacher = RSSamVisionEncoder(
    hf_pretrain_name=hf_sam_pretrain_name,
    extra_config=dict(output_hidden_states=True),
    init_cfg=dict(type='Pretrained', checkpoint=hf_sam_pretrain_ckpt_path)
).to(device)

# 定义剪枝函数
def weight_based_pruning(model, amount=0.2):
    """
    对 ViT 模型的权重进行剪枝
    :param model: 训练好的 ViT 模型
    :param amount: 剪枝比例，例如 0.2 表示剪枝 20% 的权重
    """
    for name, module in model.named_modules():
        # 对全连接层（Linear Layers）进行剪枝
        if isinstance(module, nn.Linear):
            # 使用 L1 范数剪枝
            prune.l1_unstructured(module, name='weight', amount=amount)
            # 永久移除剪枝的权重
            prune.remove(module, 'weight')

# 对模型进行剪枝
weight_based_pruning(teacher, amount=0.2)

# 检查剪枝后的模型权重
print("剪枝后的权重：")
for name, param in teacher.named_parameters():
    if 'weight' in name:
        print(f"{name}: {torch.sum(param != 0).item()} / {param.numel()} 非零权重")

# 保存剪枝后的模型权重
output_ckpt_path = "work_dirs/sam_cache/sam_vit_base/pruned_pytorch_model.bin"
torch.save(teacher.state_dict(), output_ckpt_path)
print(f"剪枝后的模型权重已保存到: {output_ckpt_path}")