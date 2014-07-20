#!/bin/bash

FILE=${1}
HOST=`hostname`"_1"
LOG="run2.sh.log"


cd ${2}Results
mkdir ${HOST}
cd ${HOST}

IFS_BACKUP=${IFS}
IFS='.'
array=( $FILE )
NAME=${array[0]}


IFS=${IFS_BACKUP}

for ((id=1; id<=3; id=id+1))
do
  LOG_FILE1="${NAME}-${id}rawClassificationResults.txt"
  LOG_FILE2="${NAME}-${id}GTMClassificationResults.txt"
  python ../../rawClassification.py "../../Data/${FILE}"  &> ${LOG_FILE1}
  echo "RAW DONE" > ${LOG}
  python ../../GTMClassification.py "../../Data/${FILE}"  &> ${LOG_FILE2}
  echo "GTM DONE" > ${LOG}
done

