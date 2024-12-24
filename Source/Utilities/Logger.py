import os
import datetime

class Logger:
	def __init__(self, file):
		try:
			self.file_log = open(file, "a")
		except:
			print("Can't open file!")

	def log(self, message, is_printing=False):
		message = f"[{datetime.datetime.now().strftime("%H:%M:%S")}] {message}\n"
		if is_printing:
			print(message, end="")
		self.file_log.write(message)
		self.file_log.flush()