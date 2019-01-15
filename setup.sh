THISDIR=$(cd $(dirname $BASH_SOURCE); pwd)

source activate py36tensorflow

export PYTHONPATH
echo $PYTHONPATH | grep -q "$THISDIR/lib:" || PYTHONPATH=$THISDIR/lib:$PYTHONPATH
export PREFERRED_GPUS=4,5,6,7
