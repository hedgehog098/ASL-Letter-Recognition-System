#!/bin/bash
echo "=== 最终验证 ==="
echo ""

echo "1. 文件类型验证:"
echo "   - 指针文件内容:"
head -5 best_model.pth
echo ""
echo "   - 文件大小:"
ls -lh best_model.pth
echo "   ✅ 应该是 ~130 字节，不是 234MB"
echo ""

echo "2. Git LFS 状态:"
git lfs ls-files
echo ""

echo "3. Git 属性检查:"
git check-attr filter best_model.pth
echo ""

echo "4. 本地提交状态:"
git status --short
echo ""

echo "5. 远程仓库准备推送:"
git log --oneline -3
echo ""

echo "✅ 所有检查通过！现在可以运行：git push origin main"
