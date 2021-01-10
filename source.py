from __future__ import division
import parse, data_handle, exec_handle, result_recorder





def main():
	
	# Parse user input
	parser = parse.Parser() 
	params = parser.parse_input()


	# Read the data, process it, and normalize the appropriate columns
	data_handler = data_handle.Data(params.filepath, params.init)
	data = data_handler.get_data()

	# Execute k means clustering
	executor = exec_handle.Executor(data, params)
	executor.run()

	# Record results
	data = executor.get_data()
	result_rec = result_recorder.Recorder(data)
	result_rec.record('output.csv')

	print('Finished!')









if __name__ == '__main__':
	main()