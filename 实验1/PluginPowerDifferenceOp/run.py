#!/usr/bin/python
# -*- coding: utf-8 -*-
import re
import subprocess
sum1 = 0
sum2 = 0
for _ in range(50):
	child = subprocess.check_output("./power_diff_test") # 单位是秒
	# print(str(child))
	value1_str = re.search(r"(?<=compute data cost time ).*?(?= ms)", child).group() # 提取时间
	value1 = float(value1_str)
	sum1 += value1
	value2_str = re.search(r"(?<=get data cost time ).*?(?= ms)", child).group()
	value2 = float(value2_str)
	sum2 += value2
print str(child) # 输出最后一个
print("compute_time", sum1 / 50)
print("data_time", sum2 / 50)