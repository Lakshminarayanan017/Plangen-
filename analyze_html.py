import re
with open(r'c:\Users\Welcome\Desktop\PlanGen\frontend\index.html', 'r', encoding='utf-8') as f:
    text = f.read()

matches = re.findall(r'<template id=\"(.*?)\">(.*?)</template>', text, re.IGNORECASE | re.DOTALL)
for tid, content in matches:
    print(f'Template {tid}: {len(content)} bytes')
