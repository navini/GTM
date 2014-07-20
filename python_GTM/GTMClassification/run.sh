#!/bin/bash

MYDIR="."

cd plots
SUBS=`ls -l $MYDIR | egrep '^d' | awk '{print $9}'`

# "ls -l $MYDIR"      = get a directory listing
# "| egrep '^d'"           = pipe to egrep and select only the directories
# "awk '{print $8}'" = pipe the result from egrep to awk and print only the 8th field

# and now loop through the directories:
for SUB in ${SUBS}
do
	echo "IN SUB LOOP ${SUB}"
	pwd
	cd ${SUB}/Data
	pwd
	FILES=$(find . -name "*-smoothed.txt") 
	echo "FILES:${FILES}"
	for FILE in ${FILES}
	do
		echo "IN FILE LOOP  ${FILE}"
		IFS_BACKUP=${IFS}
		IFS='.'
		array=( $FILE )
		NAME=${array[1]}
		echo ${NAME}
		IFS='/'
		array=( $NAME )
		FNAME=${array[1]}
		echo "*********$SUB,...${FILE}"
		for ((id=1; id<=3; id=id+1))
		do
			echo "IN ID LOOP ${id}"
			echo "### ${id}...${FILE}"
			LOG_FILE1="../Results${NAME}-${id}rawClassificationResults.txt"
			LOG_FILE2="../Results${NAME}-${id}GTMClassificationResults.txt"
			python ../../../rawClassification.py "${FNAME}.txt"  &> ${LOG_FILE1} 
			echo "RAW DONE"
			python ../../../GTMClassification.py "${FNAME}.txt"  &> ${LOG_FILE2} 
                  		
		done
		IFS=${IFS_BACKUP}
	done
	cd ../..
done
