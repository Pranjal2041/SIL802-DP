#!/bin/bash
# This script is used to run SQL scripts on the server

# store current directory in a variable
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
rm -f $DIR/sql_output.txt
# change to the directory where the script is located
cd "/home/$USER/sil/differential-privacy/examples/zetasql"
source "/home/$USER/.bashrc"
echo "Starting Query"
echo $1
echo $2
echo $3
echo $4


bazel run execute_query -- --data_set=$(pwd)/data/$2 --userid_col="$3" $1 > $DIR/sql_output.txt