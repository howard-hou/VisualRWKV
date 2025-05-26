for f in *.PDF; do
  # 获取文件名前13个字符（格式如 2001_000009）
  prefix=${f:0:11}
  
  # 检查是否匹配 4位年份 + '_' + 6位数字
  if [[ $prefix =~ ^[0-9]{4}_[0-9]{6}$ ]]; then
    # 构建新文件名（统一小写扩展名）
    newname="${prefix}.pdf"
    mv "$f" "$newname"
  else
    # 删除不匹配的文件
    rm "$f"
  fi
done

