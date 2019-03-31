import numpy as np
import json
import os
import pandas as pd
import copy
import string
import emoji
import csv
import random

class convert(object):
	DATA_FILENAME="data.dat"
	DATA_FILENAME_EXCEL = "final-data-set.xlsx"
	DICT_FILENAME='dict.json'
	OUTPUT_FILENAME = 'output.csv'
	DICT = {}
	SAVE_COUNT = 1
	LOWER_CASE = list(string.ascii_lowercase)
	UPPER_CASE = list(string.ascii_uppercase)
	LETTERS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
	SMILES =  ['😂','😍', '🐷', '🐖', '🐽', '🔰', '🤔', '👉','👌','🔫','🖕','😇','😈','󾌾','😳','😹','😁','😤','😡','😔','🏃']
	SMILES2 =  ['‍♂','💪','😒','😕','😖','😺','❤','💕','😘','💔','😭','😅','😶','🏼']
	EMOJI = list(emoji.UNICODE_EMOJI.keys())
	SPECIAL = ['…','⁉️',]
	PUNC = ['\"', '?', '.', '!', '(', ')', ',', '\'', '”', '“', '-', '_',  '~', '=', '\\', '/',  ':', '—', ' ', ' ', '󾌾', ' ']
	CONTAIN_STRINGS_REMOVE = PUNC + LOWER_CASE + UPPER_CASE + SPECIAL + LETTERS  + EMOJI + SMILES + SMILES2
	REMOVE_WORDS_STARTING = ['@','#','http']
	CONTAIN_STRINGS_REMOVE_AFTER = ['#', '@']
	simplify_characters_dict = {
		# Consonant
		"ඛ": "ක",
		"ඝ": "ග",
		"ඟ": "ග",
		"ඡ": "ච",
		"ඣ": "ජ",
		"ඦ": "ජ",
		"ඤ": "ඥ",
		"ඨ": "ට",
		"ඪ": "ඩ",
		"ණ": "න",
		"ඳ": "ද",
		"ඵ": "ප",
		"භ": "බ",
		"ඹ": "බ",
		"ශ": "ෂ",
		"ළ": "ල",

		# Vowels
		"ආ": "අ",
		"ඈ": "ඇ",
		"ඊ": "ඉ",
		"ඌ": "උ",
		"ඒ": "එ",
		"ඕ": "ඔ",

		"ා": "",
		"ෑ": "ැ",
		"ී": "ි",
		"ූ": "ු",
		"ේ": "ෙ",
		"ෝ": "ො",
		"ෲ": "ෘ"
	}

	def main(self):
		# # TODO First time
		# self.load_dict()
		# self.create_dict()

		# #TODO Tag now
		# self.load_dict()
		# self.tag()

		#TODO Recreate Singlish paragraphs
		self.load_dict()
		self.convert_to_singlish()



	def write_to_csv(self, write_this_list):
		with open(convert.OUTPUT_FILENAME, 'a') as writeFile:
			writer = csv.writer(writeFile)
			writer.writerow(write_this_list)
		writeFile.close()

	def get_random_word(self, list_of_words):
		length_of_list = len(list_of_words)
		if length_of_list == 1:
			return list_of_words[0]
		else:
			return list_of_words[random.randint(0,length_of_list-1)]

	def convert_to_singlish(self):
		content, tags = self.load_data()
		for i, line in enumerate(content):
			print(line)
			words = self.pre_process(line)
			for word in words:
				if word in convert.DICT.keys():
					if not convert.DICT[word]['singlish'] == []:
						line = line.replace(word, self.get_random_word(convert.DICT[word]['singlish']))

			print(line, tags[i])
			self.write_to_csv([line, tags[i]])

	def get_simplified_character(self, character: str) -> str:
		if len(character) != 1:
			raise TypeError("character should be a string with length 1")
		try:
			return convert.simplify_characters_dict[character]
		except KeyError:
			return character

	def simplify_sinhalese_text(self, text: str) -> str:
		"""
        simplify
        :param text:
        :return:
        """
		modified_text = ""
		for c in text:
			modified_text += self.get_simplified_character(c)
		return modified_text


	def load_data(self):
		df = pd.read_excel(convert.DATA_FILENAME_EXCEL)
		lines = []
		tags = []
		for line in df[df.columns[1]]:
			lines.append(line)
		for tag in df[df.columns[3]]:
			tags.append(tag)

		return np.array(lines), np.array(tags)

	def tag(self):
		keys_not_tagged = [key for key in convert.DICT if len(convert.DICT[key]["singlish"]) == 0]
		total_keys_not_tagged = len(keys_not_tagged)
		total_keys_in_dict = len(convert.DICT)
		now_tag_count = 0
		for key in keys_not_tagged:
			now_tag_count += 1
			print("Tagged: {} Not-Tagged: {} Total-words: {}".format(total_keys_in_dict - total_keys_not_tagged,
																	 total_keys_not_tagged, total_keys_in_dict))
			convert.DICT[key]["singlish"] = self.get_input(key)
			total_keys_not_tagged -= 1

			if now_tag_count%convert.SAVE_COUNT == 0:
				self.save_dict()

	def get_input(self,text):
		msg = "{} : ".format(text)

		texts = []
		try:
			text = input(msg)  # Python 3
		except:
			text = raw_input(msg)  # Python 2

		for t in text.split(','):
			texts.append(t.strip())

		return texts

	def create_dict(self):
		count_save = 0
		count_duplicate = 0
		count_total_words = 0

		# with open(convert.DATA_FILENAME, 'r') as fp:
		# 	content = fp.readlines()
		# 	content = [x.strip() for x in content]
		temp_word_list = []
		content,_ = self.load_data()
		for line in content:
			for word in self.pre_process(line):
				count_total_words += 1
				if not (word in  convert.DICT):
					print(word)
					convert.DICT[word] = {"singlish" : [], "count": 1}
					count_save += 1
				else:
					count_duplicate += 1
					if word in temp_word_list:
						convert.DICT[word]['count'] = convert.DICT[word]["count"]+1
					else:
						temp_word_list.append(word)
						convert.DICT[word]['count'] = 1

		self.save_dict()
		print("total words:{} Saved:{} Duplicate:{}".format(count_total_words, count_save, count_duplicate))

	def remove_words_starting(self, word):
		flag = False
		for let_part in convert.REMOVE_WORDS_STARTING:
			if word[0:len(let_part)] == let_part:
				# print("Removed-word: {} | Coz starting: {}".format(word, let_part))
				flag = True
				break
		return flag

	def remove_by_length(self, word):
		flag = False
		if len(word) < 2:
			flag = True
		return flag

	def remove_letters_in_words(self, word, after=False):
		word_ = ''
		if not after:
			rem_words = convert.CONTAIN_STRINGS_REMOVE
		else:
			rem_words = convert.CONTAIN_STRINGS_REMOVE_AFTER
		for let in rem_words:
			if let in word:
				pre_word = copy.deepcopy(word)
				word = word.replace(let, ' ')
				# print("Altered-word: {} to {} contains: {}".format(pre_word, word, let))
		return word



	def pre_process(self, sentance):
		words = []
		for word in sentance.split():
			# print("pre-preocess:", word)

			word = self.remove_letters_in_words(word)
			if self.remove_words_starting(word):
				continue
			if self.remove_by_length(word):
				continue
			word = self.remove_letters_in_words(word,after=True)
			# word = self.simplify_sinhalese_text(word) # got it from original repo
			for _word in word.split():
				_word = _word.strip()
				words.append(_word)

		words_np = np.array(words)

		return  words_np

	def save_dict(self):
		with open(convert.DICT_FILENAME, 'w') as fp:
			json.dump(convert.DICT, fp)

	def load_dict(self):
		if os.path.exists(convert.DICT_FILENAME):
			with open(convert.DICT_FILENAME, 'r') as fp:
				convert.DICT = json.load(fp)
		else:
			print("File not found")



if __name__ == "__main__":
	con_obj = convert()
	con_obj.main()