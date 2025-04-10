import numpy as np
from datetime import datetime
import json, os
import re
# ff1,tt,zxxx1(10752, 1001)
def framepg(frame,timestamp):
	drone_contr_combine = 0
	frame = re.sub(r"^0+","",frame)
	if len(frame) < 200:
		return []
	packet_type = frame[2:6]
	if packet_type == "1002" or packet_type == "1003":
		sequence_number = frame[6:10]
		state_imformation = frame[10:14]
		serial_number = frame[14:46]
		byte_array = bytearray.fromhex(serial_number[0:6])
		serial_number = bytearray.fromhex(serial_number)
		serial_number = serial_number.decode(encoding="utf-8")
		serial_number = serial_number.strip()

		ModelPrefix = byte_array.decode()

		Drone_Longitude_GPS = frame[52:54]+frame[50:52]+frame[48:50]+frame[46:48]
		Drone_Longitude_GPS = int(Drone_Longitude_GPS,16)*180/np.pi/1e7
		if Drone_Longitude_GPS > 180:
			Drone_Longitude_GPS = Drone_Longitude_GPS - 24608.3499208
		
		Drone_Latitude_GPS = frame[60:62]+frame[58:60]+frame[56:58]+frame[54:56]
		Drone_Latitude_GPS = int(Drone_Latitude_GPS,16)*180/np.pi/1e7
		if Drone_Latitude_GPS > 90:
			Drone_Latitude_GPS = Drone_Latitude_GPS - 24608.3499208
		

		Altitude = frame[64:66]+frame[62:64]
		if int(Altitude,16)>32768:
			Altitude=-(65535-int(Altitude,16))
		else:
			Altitude = int(Altitude,16)

		Height = frame[68:70]+frame[66:68]
		if int(Height,16)>32768:
			Height=-(65535-int(Height,16))/10
		else:
			Height = int(Height,16)/10

		x_Speed = frame[72:74]+frame[70:72]
		if int(x_Speed,16)>32768:
			x_Speed=-(65535-int(x_Speed,16))/100
		else:
			x_Speed = int(x_Speed,16)/100

		y_Speed = frame[76:78]+frame[74:76]
		if int(y_Speed,16)>32768:
			y_Speed=-(65535-int(y_Speed,16))/100
		else:
			y_Speed = int(y_Speed,16)/100

		z_Speed = frame[80:82]+frame[78:80]
		if int(z_Speed,16)>32768:
			z_Speed=-(65535-int(z_Speed,16))/100
		else:
			z_Speed = int(z_Speed,16)/100

		# Yaw_Angle = frame[84:86]+frame[82:84]
		# if int(Yaw_Angle,16)>32768:
		# 	Yaw_Angle=-(65535-int(Yaw_Angle,16))/100
		# else:
		# 	Yaw_Angle=int(Yaw_Angle,16)/100 
		# if Yaw_Angle==0:
		# 	Yaw_Angle=0
		# elif Yaw_Angle<0:
		# 	Yaw_Angle=Yaw_Angle+360
		# else:
		# 	Yaw_Angle=np.mod(Yaw_Angle,180)
		
		Pilot_GPS_Clock=frame[100:102]+frame[98:100]+frame[96:98]+frame[94:96]+frame[92:94]+frame[90:92]+frame[88:90]+frame[86:88]
		Pilot_GPS_Clock=int(Pilot_GPS_Clock,16)/1000
		# Pilot_GPS_Clock=datestr((Pilot_GPS_Clock+3600*8)/86400+719529,30)

		Pilot_Latitude_GPS = frame[108:110]+frame[106:108]+frame[104:106]+frame[102:104]
		Pilot_Latitude_GPS = int(Pilot_Latitude_GPS,16)*180/np.pi/1e7		
		if Pilot_Latitude_GPS > 90:
			Pilot_Latitude_GPS = Pilot_Latitude_GPS - 24608.3499208
		Pilot_Longitude_GPS = frame[116:118]+frame[114:116]+frame[112:114]+frame[110:112]
		Pilot_Longitude_GPS = int(Pilot_Longitude_GPS,16)*180/np.pi/1e7
		if Pilot_Longitude_GPS > 180:
			Pilot_Longitude_GPS = Pilot_Longitude_GPS - 24608.3499208
		Home_Longitude_GPS = frame[124:126]+frame[122:124]+frame[120:122]+frame[118:120]
		Home_Longitude_GPS = int(Home_Longitude_GPS,16)*180/np.pi/1e7		
		if Home_Longitude_GPS > 180:
			Home_Longitude_GPS = Home_Longitude_GPS - 24608.3499208
		Home_Latitude_GPS = frame[132:134]+frame[130:132]+frame[128:130]+frame[126:128]
		Home_Latitude_GPS = int(Home_Latitude_GPS,16)*180/np.pi/1e7
		if Home_Latitude_GPS > 90:
			Home_Latitude_GPS = Home_Latitude_GPS - 24608.3499208

		modelID = frame[134:136]
		modelID = int(modelID,16)

		if modelID == 36:
			Model = "Phantom 4 Pro V2, ModelID=36"
			Modelnew = "Phantom 4 Pro V2"
		elif modelID == 58:
			Model = "Mavic Air 2, ModelID=58"
			Modelnew = "Mavic Air 2"
		elif modelID == 73:
			Model = "Mini 3 Pro, ModelID=73"
			Modelnew = "Mini 3 Pro"
		elif modelID == 1:
			Model = "Inspire 1, ModelID=1"
			Modelnew = "Inspire 1"
		elif modelID == 2:
			Model = "Phantom 3 Series, ModelID=2"
			Modelnew = "Phantom 3 Series"
		elif modelID == 3:
			Model = "Phantom 3 Series Pro, ModelID=3"
			Modelnew = "Phantom 3 Series Pro"
		elif modelID == 4:
			Model = "Phantom 3 Std, ModelID=4"
			Modelnew = "Phantom 3 Std"
		elif modelID == 5:
			Model = "M100, ModelID=5"
			Modelnew = "M100"
		elif modelID == 6:
			Model = "ACEONE, ModelID=6"
			Modelnew = "ACEONE"
		elif modelID == 7:
			Model = "WKM, ModelID=7"
			Modelnew = "WKM"
		elif modelID == 8:
			Model = "NAZA, ModelID=8"
			Modelnew = "NAZA"
		elif modelID == 9:
			Model = "A2, ModelID=9"
			Modelnew = "A2"
		elif modelID == 10:
			Model = "A3, ModelID=10"
			Modelnew = "A3"
		elif modelID == 11:
			Model = "Phantom 4, ModelID=11"
			Modelnew = "Phantom 4"
		elif modelID == 12:
			Model = "MG1, ModelID=12"
			Modelnew = "MG1"
		elif modelID == 14:
			Model = "M600, ModelID=14"
			Modelnew = "M600"
		elif modelID == 15:
			Model = "Phantom 3 4k, ModelID=15"
			Modelnew = "Phantom 3 4k"
		elif modelID == 16:
			Model = "Mavic Pro, ModelID=16"
			Modelnew = "Mavic Pro"
		elif modelID == 17:
			Model = "Inspire 2, ModelID=17"
			Modelnew = "Inspire 2"
		elif modelID == 18:
			Model = "Phantom 4 Pro, ModelID=18"
			Modelnew = "Phantom 4 Pro"
		elif modelID == 20:
			Model = "N2, ModelID=20"
			Modelnew = "N2"
		elif modelID == 21:
			Model = "Spark, ModelID=21"
			Modelnew = "Spark"
		elif modelID == 23:
			Model = "M600 Pro, ModelID=23"
			Modelnew = "M600 Pro"
		elif modelID == 24:
			Model = "Mavic Air, ModelID=24"
			Modelnew = "Mavic Air"
		elif modelID == 25:
			Model = "M200, ModelID=25"
			Modelnew = "M200"
		elif modelID == 26:
			Model = "Phantom 4 Series, ModelID=26"
			Modelnew = "Phantom 4 Series"
		elif modelID == 27:
			Model = "Phantom 4 Adv, ModelID=27"
			Modelnew = "Phantom 4 Adv"
		elif modelID == 28:
			Model = "M210, ModelID=28"
			Modelnew = "M210"
		elif modelID == 30:
			Model = "M210RTK, ModelID=30"
			Modelnew = "M210RTK"
		elif modelID == 31:
			Model = "A3_AG, ModelID=31"
			Modelnew = "A3_AG"
		elif modelID == 32:
			Model = "MG2, ModelID=32"
			Modelnew = "MG2"
		elif modelID == 34:
			Model = "MG1A, ModelID=34"
			Modelnew = "MG1A"
		elif modelID == 35:
			Model = "Phantom 4 RTK, ModelID=35"
			Modelnew = "Phantom 4 RTK"
		elif modelID == 38:
			Model = "MG1P, ModelID=38"
			Modelnew = "MG1P"
		elif modelID == 40:
			Model = "MG1P-RTK, ModelID=40"
			Modelnew = "MG1P-RTK"
		elif modelID == 41:
			Model = "Mavic 2, ModelID=41"
			Modelnew = "Mavic 2"
		elif modelID == 44:
			Model = "M200 V2 Series, ModelID=44"
			Modelnew = "M200 V2 Series"
		elif modelID == 51:
			Model = "Mavic 2 Enterprise, ModelID=51"
			Modelnew = "Mavic 2 Enterprise"
		elif modelID == 53:
			Model = "Mavic Mini, ModelID=53"
			Modelnew = "Mavic Mini"
		elif modelID == 59:
			Model = "P4M, ModelID=53"
			Modelnew = "P4M"
		elif modelID == 60:
			Model = "M300 RTK, ModelID=60"
			Modelnew = "M300 RTK"
		elif modelID == 61:
			Model = "DJI FPV, ModelID=61"
			Modelnew = "DJI FPV"
		elif modelID == 63:
			Model = "Mini 2, ModelID=63"
			Modelnew = "Mini 2"
		elif modelID == 64:
			Model = "AGRAS T10, ModelID=64"
			Modelnew = "AGRAS T10"
		elif modelID == 65:
			Model = "AGRAS T30, ModelID=65"
			Modelnew = "AGRAS T30"
		elif modelID == 66:
			Model = "Air 2S, ModelID=66"
			Modelnew = "Air 2S"
		elif modelID == 67:
			Model = "M30, ModelID=67"
			Modelnew = "M30"
		elif modelID == 68:
			Model = "Mavic 3, ModelID=68"
			Modelnew = "Mavic 3"
		elif modelID == 69:
			Model = "Mavic 2 Enterprise Adv, ModelID=69"
			Modelnew = "Mavic 2 Enterprise Adv"
		elif modelID == 70:
			Model = "Mini SE, ModelID=70"
			Modelnew = "Mini SE"
		elif modelID == 240:
			Model = "YUNEEC H480, ModelID=240"
			Modelnew = "YUNEEC H480"
		elif modelID == 75:
			Model = "DJI Avata, ModelID=75"
			Modelnew = "DJI Avata"
		elif modelID == 77:
			Model = "Mavic 3 Enterprise E_T_M, ModelID=77"
			Modelnew = "Mavic 3 Enterprise E_T_M"
		elif modelID == 82:
			Model = "AGRAS T25, ModelID=82"
			Modelnew = "AGRAS T25"
		elif modelID == 83:
			Model = "AGRAS T50, ModelID=83"
			Modelnew = "AGRAS T50"
		# elif modelID == 84:
		# 	Model = "Mavic 3 series, ModelID=84"
		# 	Modelnew = "Mavic 3 series"
		elif modelID == 86:
			Model = "DJI Mavic 3 Classic, ModelID=86"
			Modelnew = "DJI Mavic 3 Classic"
		elif modelID == 87:
			Model = "DJI Mini 3, ModelID=87"
			Modelnew = "DJI Mini 3"
		elif modelID == 88:
			Model = "DJI Mini 2 SE, ModelID=88"
			Modelnew = "DJI Mini 2 SE"
		elif modelID == 90:
			Model = "Air3, ModelID=90"
			Modelnew = "Air3"
		else:
			Model = "Unknown drone"
			Modelnew = "Unknown drone"
			print(modelID)
			

		UUIDLength = frame[136:138]
		UUIDLength = int(UUIDLength,16)

		UUID = frame[138:138+UUIDLength*2]

		else_part = frame[138+UUIDLength*2:len(frame)]

		vspeed=np.sqrt(float(x_Speed)*float(x_Speed)+float(y_Speed)*float(y_Speed))
		if float(x_Speed)==0:
			if float(y_Speed)==0:
				anglereal = 90
			else:
				anglereal = 270
		else:
			if float(x_Speed)<0:
				anglereal = np.arctan(float(y_Speed)/float(x_Speed))/np.pi*180+180	
			else:
				anglereal = np.arctan(float(y_Speed)/float(x_Speed))/np.pi*180
		if anglereal < 0:
			anglereal = anglereal+360
		else:
			anglereal = anglereal%360

		if int(Pilot_Longitude_GPS) == 0 and int(Pilot_Latitude_GPS) == 0:
			P_Longitude_GPS = Home_Longitude_GPS
			P_Latitude_GPS = Home_Latitude_GPS
		else:
			P_Longitude_GPS = Pilot_Longitude_GPS
			P_Latitude_GPS = Pilot_Latitude_GPS
		# print("\n")
		print("\n","="*10, " decode information ","="*10)
		print("dronemodel =",Modelnew)
		pilotmodel = Modelnew+"CONTR"
		print("pilotmodel =",pilotmodel)
		print("Drone_Latitude_GPS =",Drone_Latitude_GPS)
		print("Drone_Longitude_GPS =",Drone_Longitude_GPS)
		print("Pilot_Latitude_GPS =",Pilot_Latitude_GPS)
		print("Pilot_Longitude_GPS =",Pilot_Longitude_GPS)
		print("Home_Latitude_GPS =",Home_Latitude_GPS)
		print("Home_Longitude_GPS =",Home_Longitude_GPS)
		print("UUID =",serial_number)
		print("timeNow = ",timestamp)
		print("="*40)
		# print("\n")
		dt_object = datetime.fromtimestamp(int(timestamp))
		dt_objectstr=str(dt_object)
		dt_objectstr=dt_objectstr.replace(' ', '_').replace(':', '-')
		# print(dt_objectstr)
		# path = f'drone/{Modelnew}{dt_objectstr}.txt'
		current_directory = os.getcwd()
		script_directory = os.path.dirname(os.path.abspath(__file__))
		os.chdir(script_directory)
		folder_name = "drone"
		folder_path = os.path.join(script_directory, folder_name)
		if not os.path.exists(folder_path):
			os.makedirs(folder_path)
		file_name = f"{Modelnew}{dt_objectstr}.txt"
		file_path = os.path.join(folder_path, file_name)		
		
		f = open(file_path, 'w')
		f.write(f'sequence_number={sequence_number}\n')
		f.write(f'state_imformation={state_imformation}\n')
		f.write(f'serial_number={serial_number}\n')
		f.write(f'ModelPrefix={ModelPrefix}\n')
		f.write(f'Drone_GPS={Drone_Longitude_GPS},{Drone_Latitude_GPS}\n')
		f.write(f'Altitude={Altitude}\n')
		f.write(f'Height={Height}\n')
		f.write(f'x_y_z_Speed={x_Speed},{y_Speed},{z_Speed}\n')
		f.write(f'Pilot_GPS_Clock={Pilot_GPS_Clock}\n')
		f.write(f'Pilot_GPS={Pilot_Longitude_GPS},{Pilot_Latitude_GPS}\n')
		f.write(f'Home_GPS={Home_Longitude_GPS},{Home_Latitude_GPS}\n')
		f.write(f'modelID={modelID}\n')
		f.write(f'UUID={serial_number}\n')
		f.write(f'timeNow={timestamp}\n')
		f.close()
		package_to_return = []		
		print("drone_contr_combine", drone_contr_combine)
		if drone_contr_combine == 0:		
			print("1"*50)	
			# if P_Longitude_GPS==0 and P_Latitude_GPS==0:
			if P_Longitude_GPS!=0 and P_Latitude_GPS!=0:
				data = {
					"type": str(pilotmodel),
					"serial_number":str(serial_number),
					"latitude":str(P_Latitude_GPS),
					"longitude":str(P_Longitude_GPS),
					"height":"0",
					"speed":"0",
					"direction":"0",
					"timeStamp":str(timestamp),
					"altitude":str(Altitude)
					# "init_pos_lock":"0"
				}
				data_pilot = json.dumps(data).replace(r'\u0000', '')
				package_to_return.append(data_pilot)
			# if Drone_Longitude_GPS==0 and Drone_Latitude_GPS==0:
			if Drone_Longitude_GPS!=0 and Drone_Latitude_GPS!=0:
				data = {
					"type": str(Modelnew),
					"serial_number":str(serial_number),
					"latitude":str(Drone_Latitude_GPS),
					"longitude":str(Drone_Longitude_GPS),
					"height":str(Height),
					"speed":str(vspeed),
					"direction":str(anglereal),
					"timeStamp":str(timestamp),
					"altitude":str(Altitude)
					# "init_pos_lock":"0"
				}
				data_drone = json.dumps(data).replace(r'\u0000', '')
				print("data_drone",data_drone)
				package_to_return.append(data_drone)
			return package_to_return

		else:
			data = {
					"type": str(Modelnew),
					"serial_number":str(serial_number),
					"latitude_drone":str(Drone_Latitude_GPS),
					"longitude_drone":str(Drone_Longitude_GPS),
					"latitude_contr":str(P_Latitude_GPS),
					"longitude_contr":str(P_Longitude_GPS),
					"height_drone":str(Height),
					"speed_drone":str(vspeed),
					"direction_drone":str(anglereal),
					"height_contr":"0",
					"speed_contr":"0",
					"direction_contr":"0",
					"timeStamp":str(timestamp)
					# "Altitude":"0"
				}
			data_drone_contr = json.dumps(data).replace(r'\u0000', '')
			return [data_drone_contr]
		
	else:
		return []



if __name__ == "__main__":
	import subprocess
	# import datetime
	from datetime import datetime
	import json, os
	#returned_text = subprocess.check_output("sudo /home/iwave/mattaim/idBS/idBS/wintoubuntu/cpp/remove_turbo /home/iwave/mattaim/idBS/idBS/wintoubuntu/bits", shell=True)
	start_timestamp = datetime.now() 
	#returned_text = returned_text.decode("utf-8")
	#returned_text = returned_text.strip()
	#returned_text = returned_text.strip("0")
	timestamp = start_timestamp.timestamp()
	#print(returned_text)
	returned_text = "5810029400071f334e33424a38453031323032384400000000000000000000efff000000000000000012e0b596c7b38c010000d0f8a5ff5cd7920100000000000000003a1331313933303335383832343838383135363136006e3300000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000067be076b7327a755285995c055eb568fd482212287be2c9f4b0e71abcb8924ff944a"
	upload_data = framepg(returned_text,timestamp)
	print("upload_data",upload_data)