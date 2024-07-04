import lcd
import sensor
import gc
from board import board_info
from fpioa_manager import fm
from machine import UART
from maix import KPU
lcd.init()
sensor.reset()
sensor.set_framesize(sensor.QVGA)
sensor.set_pixformat(sensor.RGB565)
sensor.set_vflip(True)
fm.register(board_info.EX_UART1_TX, fm.fpioa.UART1_TX)
fm.register(board_info.EX_UART1_RX, fm.fpioa.UART1_RX)
fm.register(board_info.EX_UART2_TX, fm.fpioa.UART2_TX)
fm.register(board_info.EX_UART2_RX, fm.fpioa.UART2_RX)
uart1 = UART(UART.UART1, 115200)
uart2 = UART(UART.UART2, 115200)
anchor = (8.30891522166988, 2.75630994889035, 5.18609903718768,
1.7863757404970702, 6.91480529053198, 3.825771881004435, 10.218567655549439,
3.69476690620971, 6.4088204258368195, 2.38813526350986)
names = []
lp_detecter = KPU()
lp_detecter.load_kmodel('/sd/KPU/lp_detect.kmodel')
lp_detecter.init_yolo2(anchor, anchor_num=len(anchor) // 2, img_w=320,
img_h=240, net_w=320, net_h=240, layer_w=20, layer_h=15, threshold=0.7,
nms_value=0.3, classes=len(names))
provinces = ['Wan', 'Hu', 'Jin', 'Yu^', 'Ji', 'Sx', 'Meng', 'Liao', 'Jl', 'Hei',
'Su', 'Zhe', 'Jing', 'Min', 'Gan', 'Lu', 'Yu', 'E^', 'Xiang', 'Yue', 'Gui^',
'Qiong', 'Cuan', 'Gui', 'Yun', 'Zang', 'Shan', 'Gan^', 'Qing', 'Ning', 'Xin']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P',
'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5',
'6', '7', '8', '9']
lp_recognizer = KPU()
lp_recognizer.load_kmodel("/sd/KPU/lp_recog.kmodel")
lp_recognizer.lp_recog_load_weight_data("/sd/KPU/lp_weight.bin")
def extend_box(x, y, w, h, scale):
	x1 = int(x - scale * w)
	x2 = int(x + w - 1 + scale * w)
	y1 = int(y - scale * h)
	y2 = int(y + h - 1 + scale * h)
	x1 = x1 if x1 > 0 else 0
	x2 = x2 if x2 < (320 - 1) else (320 - 1)
	y1 = y1 if y1 > 0 else 0
	y2 = y2 if y2 < (240 - 1) else (240 - 1)
	return x1, y1, x2 - x1 + 1, y2 - y1 + 1
while True:
	img = sensor.snapshot()
	lp_detecter.run_with_output(img)
	lps = lp_detecter.regionlayer_yolo2()
	for lp in lps:
		x, y, w, h = extend_box(lp[0], lp[1], lp[2], lp[3], 0.08)
		img.draw_rectangle(x, y, w, h, color=(0, 255, 0))
		lp = []
		lp_img = img.cut(x, y, w, h)
		resize_img = lp_img.resize(208, 64)
		resize_img.replace(hmirror=True)
		resize_img.pix_to_ai()
		lp_recognizer.run_with_output(resize_img)
		output = lp_recognizer.lp_recog()
		for o in output:
			lp.append(o.index(max(o)))
		img.draw_string(x + 2, y - 20 - 2, '%s %s-%s%s%s%s%s' %(provinces[lp[0]],
ads[lp[1]], ads[lp[2]], ads[lp[3]], ads[lp[4]], ads[lp[5]], ads[lp[6]]),
color=(255, 0, 0), scale=2)
		uart1.write('%s %s%s%s%s%s%s' %(provinces[lp[0]],
ads[lp[1]], ads[lp[2]], ads[lp[3]], ads[lp[4]], ads[lp[5]], ads[lp[6]]))
		uart2.write('%s %s%s%s%s%s%s' %(provinces[lp[0]],
ads[lp[1]], ads[lp[2]], ads[lp[3]], ads[lp[4]], ads[lp[5]], ads[lp[6]]))
		del lp
		del lp_img
		del resize_img
	lcd.display(img)
	gc.collect()