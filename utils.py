import os
import numpy as np

class DataLoader(object):

	def __init__(self, data_dir, batch_size, seq_length):
		self.data_dir = data_dir
		self.batch_size = batch_size
		self.seq_length = seq_length

		input_file = os.path.join(data_dir, "input.txt")
		vocab_file = os.path.join(data_dir, "vocab.pkl")
		tensor_file = os.path.join(data_dir, "data.npy")

		'''
		if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
			print("reading text file")
			self.preprocess(input_file, vocab_file, tensor_file)
		else:
			print("loading preprocessed files")
			self.load_preprocessed(vocab_file, tensor_file)
		'''

		self.preprocess(input_file, vocab_file, tensor_file)
		self.create_batches()
		self.reset_batch_pointer()	

	def get_xy(self):
		# data accessor for sklearn model
		# ratio: ratio for train - test
		x = [c for c,h in self.data ]
		y = self.tensor.T[1]

		return x,y

	def preprocess( self, input_file, vocab_file, tensor_file):
		self.data = data = []
		alpha = set()
		hashtags_lib = set()
		with open(input_file, "r") as f:
			raw = f.read()
			lines = raw.split('\n')[:-1]
			for line in lines:
				parts = line.split('#')
				content = parts[0]
				hashtags = [ h.strip() for h in parts[1].split(',')]
				data += [[content, hashtags]]

				# track vocab
				for c in content:
					alpha.add(c)

				# track hashtags
				for h in hashtags:
					hashtags_lib.add(h)

		# Map alpha, hash indexes
		self.i_alpha = dict(enumerate(alpha))
		self.i_hash = dict(enumerate(hashtags_lib))
		self.alpha_i = { self.i_alpha[k]:k for k in self.i_alpha }
		self.hash_i = { self.i_hash[k]:k for k in self.i_hash }
		self.num_alpha = len(self.alpha_i)
		self.num_hash = len(self.hash_i)

		# Build content and hashtag vectors
		self.tensor = []
		for i in range(len(data)):
			content = data[i][0]
			c_vec = []
			for c in content:
				# convert each char to a one hot
				c_i = self.alpha_i[c]
				c_vec += [self.i_hot(c_i,self.num_alpha)]

			h = data[i][1][0]
			self.tensor += [[c_vec,self.hash_i[h]]]

		self.tensor = np.array(self.tensor)

	def i_hot(self, i, s):
		# convert index to one hot
		hot = np.zeros(s)
		hot[i] = 1
		return hot

	def create_batches(self):
		self.num_batches = int(self.tensor.shape[0] / self.batch_size)

		# When the data (tensor) is too small, let's give them a better error message
		if self.num_batches==0:
			assert False, "Not enough data. Make seq_length and batch_size small."
		
		cropped = self.tensor[:self.num_batches * self.batch_size]
		xdata = cropped.T[0]
		ydata = cropped.T[1]
		self.x_batches = np.split(xdata, self.num_batches)
		self.y_batches = np.split(ydata, self.num_batches)

	def next_batch(self):
		x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
		self.pointer += 1
		return x, y

	def reset_batch_pointer(self):
		self.pointer = 0	

if __name__ == '__main__':
	loader = DataLoader('data/tweets',5,144)		
	x,y = loader.get_xy()
	print(len(x), len(y))
	
