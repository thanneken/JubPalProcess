#!/usr/bin/env bash
if [[ $# -eq 4 ]] ; then
	rfile=$(find ../ -name $1)
	echo "Found $rfile"
	gfile=$(find ../ -name $2)
	echo "Found $gfile"
	bfile=$(find ../ -name $3)
	echo "Found $bfile"
	outfile=$4
	basename="${outfile:0:(-4)}"
	extension="${outfile:(-3)}"
	if [[ "$extension" == 'tif' ]] ; then
		echo "Creating ${basename}-RGB.${extension}"
		convert "$rfile" "$gfile" "$bfile" -depth 8 -channel RGB -combine -set comment "$1\n$2\n$3\n" "${basename}-RGB.${extension}"
		echo "Creating ${basename}-RBG.${extension}"
		convert "$rfile" "$bfile" "$gfile" -depth 8 -channel RGB -combine -set comment "$1\n$3\n$2\n" "${basename}-RBG.${extension}"
		echo "Creating ${basename}-GRB.${extension}"
		convert "$gfile" "$rfile" "$bfile" -depth 8 -channel RGB -combine -set comment "$2\n$1\n$3\n" "${basename}-GRB.${extension}"
		echo "Creating ${basename}-GBR.${extension}"
		convert "$gfile" "$bfile" "$rfile" -depth 8 -channel RGB -combine -set comment "$2\n$3\n$1\n" "${basename}-GBR.${extension}"
		echo "Creating ${basename}-BRG.${extension}"
		convert "$bfile" "$gfile" "$rfile" -depth 8 -channel RGB -combine -set comment "$3\n$2\n$1\n" "${basename}-BRG.${extension}"
		echo "Creating ${basename}-BGR.${extension}"
		convert "$bfile" "$rfile" "$gfile" -depth 8 -channel RGB -combine -set comment "$3\n$1\n$2\n" "${basename}-BGR.${extension}"
	elif [[ "$extension" == 'jpg' ]] ; then
		echo "Creating ${basename}-RGB.${extension}"
		convert "$rfile" "$gfile" "$bfile" -channel RGB -combine -set comment "$1\n$2\n$3\n" "${basename}-RGB.${extension}"
		echo "Creating ${basename}-RBG.${extension}"
		convert "$rfile" "$bfile" "$gfile" -channel RGB -combine -set comment "$1\n$3\n$2\n" "${basename}-RBG.${extension}"
		echo "Creating ${basename}-GRB.${extension}"
		convert "$gfile" "$rfile" "$bfile" -channel RGB -combine -set comment "$2\n$1\n$3\n" "${basename}-GRB.${extension}"
		echo "Creating ${basename}-GBR.${extension}"
		convert "$gfile" "$bfile" "$rfile" -channel RGB -combine -set comment "$2\n$3\n$1\n" "${basename}-GBR.${extension}"
		echo "Creating ${basename}-BRG.${extension}"
		convert "$bfile" "$gfile" "$rfile" -channel RGB -combine -set comment "$3\n$2\n$1\n" "${basename}-BRG.${extension}"
		echo "Creating ${basename}-BGR.${extension}"
		convert "$bfile" "$rfile" "$gfile" -channel RGB -combine -set comment "$3\n$1\n$2\n" "${basename}-BGR.${extension}"
	fi
	echo "Command to view component files: identify -format %c <filename>"
else 
	echo "Command takes four arguments: red filename, green filename, blue filename, RGB output filename"
fi
