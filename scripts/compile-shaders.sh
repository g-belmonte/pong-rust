input_folder="$1"
output_folder="$2"
  
# Check if glslc exists
if ! [[ -x "$(command -v glslc)" ]]; then
	echo "Error: glslc was not found on \$PATH."
	exit 1
fi
	 
# Check if input folder exists
if [[ ! -d "$input_folder" ]]; then
	echo "Error: Input folder does not exist."
	exit 1
fi
	 
# Check if output folder exists, if not create it
if [[ ! -d "$output_folder" ]]; then
	mkdir -p "$output_folder"
else
	rm -rf "$output_folder/*"
fi
	 
# Execute command on each file in the input folder
for file in "$input_folder"/*; do
	# Get the filename without the path
	filename=$(basename "$file")
	 
	# Execute the command on the file and save the output in the output folder
	glslc "$file" -o "$output_folder/$filename.spv"
done
	 