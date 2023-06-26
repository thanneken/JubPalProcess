#!/usr/bin/env bash

if [[ $# -eq 4 ]] ; then
	rfile=$(find ../ -name $1)
	echo "Found $rfile"
	gfile=$(find ../ -name $2)
	echo "Found $gfile"
	bfile=$(find ../ -name $3)
	echo "Found $bfile"
	outfile=$4
	# convert "$rfile" "$gfile" "$bfile" -depth 8 -channel RGB -combine -set comment "$1\n$2\n$3\n" "$outfile"
	convert "$rfile" "$gfile" "$bfile" -channel RGB -combine -set comment "$1\n$2\n$3\n" "1-$outfile"
	convert "$rfile" "$bfile" "$gfile" -channel RGB -combine -set comment "$1\n$3\n$2\n" "2-$outfile"
	convert "$gfile" "$rfile" "$bfile" -channel RGB -combine -set comment "$2\n$1\n$3\n" "3-$outfile"
	convert "$gfile" "$bfile" "$rfile" -channel RGB -combine -set comment "$2\n$3\n$1\n" "4-$outfile"
	convert "$bfile" "$gfile" "$rfile" -channel RGB -combine -set comment "$3\n$2\n$1\n" "5-$outfile"
	convert "$bfile" "$rfile" "$gfile" -channel RGB -combine -set comment "$3\n$1\n$2\n" "6-$outfile"
	echo "Command to view component files: identify -format %c <filename>"
else 
	echo "Command takes four arguments: red filename, green filename, blue filename, RGB output filename"
fi
