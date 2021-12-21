
#source ../../scripts/env_def.sh

function convert_yolov5_pytorchnet()
{
  local transform_op=(
    '\ \ transform_param {'
    '\ \ \ \ transform_op {'
    '\ \ \ \ \ \ op: STAND'
    '\ \ \ \ \ \ scale: 0.003921569'
    '\ \ \ \ \ \ bgr2rgb: true'
    '\ \ \ \ }'
    '\ \ }'
  )
  python3 -m ufw.tools.pt_to_umodel -m $1/models/$2_jit.pt -s [1,3,640,640] -D /data/yolov5_640640_lmdb -d $1/compilation
  ret=$?
  if [ "$ret" != "0" ]; then
    return $ret
  fi

  for (( line=${#transform_op[*]}; line>0; line--))
  do
    sed -i "17i${transform_op[(($line-1))]}" $1/compilation/$2_jit_bmnetp_test_fp32.prototxt
  done

}

function quant_yolov5_net()
{
  $CALIBRATION_BIN quantize \
    --model=./compilation/$1_jit_bmnetp_test_fp32.prototxt --weights=./compilation/$1_jit_bmnetp.fp32umodel \
    --iterations=200 -save_test_proto=true --bitwidth=TO_INT8 -debug_log_level=0 -graph_transform -fpfwd_outputs="< 24 >14,< 24 >51,< 24 >82" -accuracy_opt
  ret=$?
  return $ret
}

# this is only a record of the command, use ufw test here, don't need to convert to bmodel
function convert_yolov5_to_bmodel()
{
  bmnetu -model ./compilation/$1_jit_bmnetp_deploy_int8_unique_top.prototxt -weight ./compilation/$1_jit_bmnetp.int8umodel -prec INT8 -shapes "[1,3,640,640]" --cmp=true -v=4
}
