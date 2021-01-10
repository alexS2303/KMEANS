import argparse
# User data parsing module
# To be called only by main.py


class Parser:
	def __init__(self):
		self.parser = argparse.ArgumentParser(prog='python main.py', description='Movie clustering software')
		print('\n preprocessing...\n')


	# Parsing function
	# returns an args object that has user input as properties
	def parse_input(self):
		parser = self.parser


		parser.add_argument('filepath', metavar='movies_file_path', type=str, help='path to moves.csv')
		parser.add_argument('k', metavar='k', type=int, help='The number of desired clusters')
		parser.add_argument('init', metavar='init_condition', type=str, help='initalization type (default: random)')
		

		return parser.parse_args()


