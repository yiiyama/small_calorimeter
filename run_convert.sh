#!/bin/bash

# submit with
# condor-run run_convert.sh -a inputs.list -x ~/x509up_u51268

PROD=prod8

CONVERTER=$1
INPUT=$2

cd $HOME/cmssw/CMSSW_10_2_5_SL6/
eval `scram runtime -sh`
cd -

$HOME/src/small_calorimeter/bin/convert.py $CONVERTER $INPUT out.tfrecords

export X509_USER_PROXY=$PWD/x509up_u51268

xrdcp out.tfrecords root://eoscms.cern.ch//store/user/yiiyama/small_calorimeter/all_classes/$PROD/$CONVERTER/$(basename $INPUT | sed 's/[.]root/.tfrecords/')
