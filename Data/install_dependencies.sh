#!/bin/bash
# 依赖安装脚本

echo "开始安装依赖..."
pip install -r requirements.txt --break-system-packages
echo "依赖安装完成！"
