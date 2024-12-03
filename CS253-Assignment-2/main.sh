#!/usr/bin/sh

#Checking if user has given correct number of inputs
if [ $# -eq 2 ]
then 
#Checking weather read and write is not done on same file
	if [ $1 =  $2 ]
	then 
		echo same input output file !!!!
	else
	#	Checking if input file exists

		if test -a $1
		then
			touch text1.txt
			rm text1.txt
		else
			echo "No such input file exists"
		fi

		if test -a $1
		then 
	       		outputFile="$2"
			inputFile="$1"
			touch text1.txt
		#	Creating a temperoary file to store input data without headers
                        awk -F "," '{if (NR !=1) {print $0 }}' $1 >> text1.txt
                        echo ---------------------------- > "$outputFile"
			echo " " >> "$outputFile"
			echo Unique cities in given data file: >>"$outputFile"
			#This awk command first extracts all the data in third column
			#sort will sort the data alphabetically
			#uni will return only unique data
                        awk -F "," '{print $3}' text1.txt  | sort | uniq >> "$outputFile"
			echo ---------------------------- >> "$outputFile"
			echo " " >> "$outputFile"
			echo Details of top 3 individuals with the highest salary:>>"$outputFile"
			#Creating three temproary files to store extracted data after each operation
			touch temp.txt
			touch temp2.txt
			 touch temp3.txt
			 #temp.txt stores the data from text1.txt
			 cat  text1.txt >> temp.txt
			 #temp2.txt stores the sorted data in ascending order  according to  fourth column
			  sort -t"," -k4 temp.txt> temp2.txt 
			  #temp3.txt stores the data in temp2.txt in reverse order
			  sort -t"," -k4 -r temp2.txt>temp3.txt
                         # This awk command stores the top three entries in temp3.txt
			 awk -F "," '{if (NR <4) {print $0 }}' temp3.txt >> "$outputFile"
	               # awk -F "," '{if (NR <4) {print $0 }}' temp3.txt
			rm temp2.txt
			rm temp3.txt
			 rm temp.txt
			 echo ---------------------------- >> "$outputFile"
			 echo " ">> "$outputFile"
                         #echo text1 removed
			rm text1.txt

			echo "Details of average salary of each city: " >> "$outputFile"

			#This grep command calculates average salary for each city and stores in output file
			grep -v '^Name' "$inputFile" | awk -F',' '{sum[$3]+=$4; count[$3]++} END {for (city in sum) print "City:", city, ", Salary:", sum[city]/count[city]}' >> "$outputFile"

			echo ---------------------------- >> "$outputFile"

			echo " " >>"$outputFile"

			#Calculating the average salary
			avg_totalSalary=$(grep -v '^Name' "$inputFile" | awk -F ', ' '{ sum += $4 } END { print sum/NR }')

			echo "Details of individuals with a salary above the overall average salary: " >> "$outputFile"

#This grep command selects the individuals with salary greater than average salary (avg_totalSalary)
grep -v '^Name' "$inputFile" | awk -F ', ' -v total_salary="$avg_totalSalary" '$4 > total_salary { print }' >> "$outputFile"
			#echo " "
		fi

	fi

else
	echo invalid number of arguments...
	echo ""
	echo usage message: please enter input in format as follows
	echo "1.input file name {space}  2. output file name" 
fi
