┬╝

▒ѓ
D
AddV2
x"T
y"T
z"T"
Ttype:
2	ђљ
B
AssignVariableOp
resource
value"dtype"
dtypetypeѕ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
Џ
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

└
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(ѕ
>
Minimum
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeѕ
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
Й
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ѕ
@
StaticRegexFullMatch	
input

output
"
patternstring
Ш
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.6.02v2.6.0-rc2-32-g919f693420e8На	
ї
decoder_input/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		@
*%
shared_namedecoder_input/kernel
Ё
(decoder_input/kernel/Read/ReadVariableOpReadVariableOpdecoder_input/kernel*&
_output_shapes
:		@
*
dtype0
|
decoder_input/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_namedecoder_input/bias
u
&decoder_input/bias/Read/ReadVariableOpReadVariableOpdecoder_input/bias*
_output_shapes
:@*
dtype0
ѓ
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:@@*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:@*
dtype0
ѓ
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
:@@*
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
:@*
dtype0
ѓ
conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_3/kernel
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*&
_output_shapes
:@@*
dtype0
r
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_3/bias
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes
:@*
dtype0
ѓ
conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv2d_4/kernel
{
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*&
_output_shapes
:@*
dtype0
r
conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_4/bias
k
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
_output_shapes
:*
dtype0

NoOpNoOp
г
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*у
valueПB┌ BМ
Ѕ
layer_with_weights-0
layer-0
	variables
trainable_variables
regularization_losses
	keras_api

signatures
╔
layer-0
layer_with_weights-0
layer-1
	layer_with_weights-1
	layer-2

layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
layer-8
	variables
trainable_variables
regularization_losses
	keras_api
F
0
1
2
3
4
5
6
7
8
9
F
0
1
2
3
4
5
6
7
8
9
 
Г
	variables
trainable_variables
regularization_losses
layer_metrics

layers
 metrics
!layer_regularization_losses
"non_trainable_variables
 
 
h

kernel
bias
#	variables
$trainable_variables
%regularization_losses
&	keras_api
h

kernel
bias
'	variables
(trainable_variables
)regularization_losses
*	keras_api

+	keras_api
h

kernel
bias
,	variables
-trainable_variables
.regularization_losses
/	keras_api
h

kernel
bias
0	variables
1trainable_variables
2regularization_losses
3	keras_api

4	keras_api
h

kernel
bias
5	variables
6trainable_variables
7regularization_losses
8	keras_api

9	keras_api
F
0
1
2
3
4
5
6
7
8
9
F
0
1
2
3
4
5
6
7
8
9
 
Г
	variables
trainable_variables
regularization_losses
:layer_metrics

;layers
<metrics
=layer_regularization_losses
>non_trainable_variables
PN
VARIABLE_VALUEdecoder_input/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEdecoder_input/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv2d_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_2/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv2d_2/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_3/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv2d_3/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_4/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv2d_4/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
 

0
 
 
 

0
1

0
1
 
Г
#	variables
$trainable_variables
%regularization_losses
?layer_metrics

@layers
Ametrics
Blayer_regularization_losses
Cnon_trainable_variables

0
1

0
1
 
Г
'	variables
(trainable_variables
)regularization_losses
Dlayer_metrics

Elayers
Fmetrics
Glayer_regularization_losses
Hnon_trainable_variables
 

0
1

0
1
 
Г
,	variables
-trainable_variables
.regularization_losses
Ilayer_metrics

Jlayers
Kmetrics
Llayer_regularization_losses
Mnon_trainable_variables

0
1

0
1
 
Г
0	variables
1trainable_variables
2regularization_losses
Nlayer_metrics

Olayers
Pmetrics
Qlayer_regularization_losses
Rnon_trainable_variables
 

0
1

0
1
 
Г
5	variables
6trainable_variables
7regularization_losses
Slayer_metrics

Tlayers
Umetrics
Vlayer_regularization_losses
Wnon_trainable_variables
 
 
?
0
1
	2

3
4
5
6
7
8
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
љ
serving_default_decoder_inputPlaceholder*/
_output_shapes
:           
*
dtype0*$
shape:           

є
StatefulPartitionedCallStatefulPartitionedCallserving_default_decoder_inputdecoder_input/kerneldecoder_input/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasconv2d_4/kernelconv2d_4/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЯЯ*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *.
f)R'
%__inference_signature_wrapper_5767533
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ю
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename(decoder_input/kernel/Read/ReadVariableOp&decoder_input/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOp!conv2d_4/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *)
f$R"
 __inference__traced_save_5768069
л
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedecoder_input/kerneldecoder_input/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasconv2d_4/kernelconv2d_4/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *,
f'R%
#__inference__traced_restore_5768109Ћр
Ч\
Р	
E__inference_decoder1_layer_call_and_return_conditional_losses_5767647

inputsX
>decoder_decoder_input_conv2d_transpose_readvariableop_resource:		@
C
5decoder_decoder_input_biasadd_readvariableop_resource:@I
/decoder_conv2d_1_conv2d_readvariableop_resource:@@>
0decoder_conv2d_1_biasadd_readvariableop_resource:@I
/decoder_conv2d_2_conv2d_readvariableop_resource:@@>
0decoder_conv2d_2_biasadd_readvariableop_resource:@I
/decoder_conv2d_3_conv2d_readvariableop_resource:@@>
0decoder_conv2d_3_biasadd_readvariableop_resource:@I
/decoder_conv2d_4_conv2d_readvariableop_resource:@>
0decoder_conv2d_4_biasadd_readvariableop_resource:
identityѕб'decoder/conv2d_1/BiasAdd/ReadVariableOpб&decoder/conv2d_1/Conv2D/ReadVariableOpб'decoder/conv2d_2/BiasAdd/ReadVariableOpб&decoder/conv2d_2/Conv2D/ReadVariableOpб'decoder/conv2d_3/BiasAdd/ReadVariableOpб&decoder/conv2d_3/Conv2D/ReadVariableOpб'decoder/conv2d_4/BiasAdd/ReadVariableOpб&decoder/conv2d_4/Conv2D/ReadVariableOpб,decoder/decoder_input/BiasAdd/ReadVariableOpб5decoder/decoder_input/conv2d_transpose/ReadVariableOpp
decoder/decoder_input/ShapeShapeinputs*
T0*
_output_shapes
:2
decoder/decoder_input/Shapeа
)decoder/decoder_input/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)decoder/decoder_input/strided_slice/stackц
+decoder/decoder_input/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+decoder/decoder_input/strided_slice/stack_1ц
+decoder/decoder_input/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+decoder/decoder_input/strided_slice/stack_2Т
#decoder/decoder_input/strided_sliceStridedSlice$decoder/decoder_input/Shape:output:02decoder/decoder_input/strided_slice/stack:output:04decoder/decoder_input/strided_slice/stack_1:output:04decoder/decoder_input/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#decoder/decoder_input/strided_sliceЂ
decoder/decoder_input/stack/1Const*
_output_shapes
: *
dtype0*
value
B :Я2
decoder/decoder_input/stack/1Ђ
decoder/decoder_input/stack/2Const*
_output_shapes
: *
dtype0*
value
B :Я2
decoder/decoder_input/stack/2ђ
decoder/decoder_input/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
decoder/decoder_input/stack/3ќ
decoder/decoder_input/stackPack,decoder/decoder_input/strided_slice:output:0&decoder/decoder_input/stack/1:output:0&decoder/decoder_input/stack/2:output:0&decoder/decoder_input/stack/3:output:0*
N*
T0*
_output_shapes
:2
decoder/decoder_input/stackц
+decoder/decoder_input/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+decoder/decoder_input/strided_slice_1/stackе
-decoder/decoder_input/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-decoder/decoder_input/strided_slice_1/stack_1е
-decoder/decoder_input/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-decoder/decoder_input/strided_slice_1/stack_2­
%decoder/decoder_input/strided_slice_1StridedSlice$decoder/decoder_input/stack:output:04decoder/decoder_input/strided_slice_1/stack:output:06decoder/decoder_input/strided_slice_1/stack_1:output:06decoder/decoder_input/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%decoder/decoder_input/strided_slice_1ш
5decoder/decoder_input/conv2d_transpose/ReadVariableOpReadVariableOp>decoder_decoder_input_conv2d_transpose_readvariableop_resource*&
_output_shapes
:		@
*
dtype027
5decoder/decoder_input/conv2d_transpose/ReadVariableOpИ
&decoder/decoder_input/conv2d_transposeConv2DBackpropInput$decoder/decoder_input/stack:output:0=decoder/decoder_input/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*1
_output_shapes
:         ЯЯ@*
paddingSAME*
strides
2(
&decoder/decoder_input/conv2d_transpose╬
,decoder/decoder_input/BiasAdd/ReadVariableOpReadVariableOp5decoder_decoder_input_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,decoder/decoder_input/BiasAdd/ReadVariableOpВ
decoder/decoder_input/BiasAddBiasAdd/decoder/decoder_input/conv2d_transpose:output:04decoder/decoder_input/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ@2
decoder/decoder_input/BiasAddц
decoder/decoder_input/ReluRelu&decoder/decoder_input/BiasAdd:output:0*
T0*1
_output_shapes
:         ЯЯ@2
decoder/decoder_input/Relu╚
&decoder/conv2d_1/Conv2D/ReadVariableOpReadVariableOp/decoder_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02(
&decoder/conv2d_1/Conv2D/ReadVariableOpЩ
decoder/conv2d_1/Conv2DConv2D(decoder/decoder_input/Relu:activations:0.decoder/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ@*
paddingSAME*
strides
2
decoder/conv2d_1/Conv2D┐
'decoder/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp0decoder_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'decoder/conv2d_1/BiasAdd/ReadVariableOp╬
decoder/conv2d_1/BiasAddBiasAdd decoder/conv2d_1/Conv2D:output:0/decoder/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ@2
decoder/conv2d_1/BiasAddЋ
decoder/conv2d_1/ReluRelu!decoder/conv2d_1/BiasAdd:output:0*
T0*1
_output_shapes
:         ЯЯ@2
decoder/conv2d_1/Relu▄
"decoder/tf.__operators__.add/AddV2AddV2#decoder/conv2d_1/Relu:activations:0(decoder/decoder_input/Relu:activations:0*
T0*1
_output_shapes
:         ЯЯ@2$
"decoder/tf.__operators__.add/AddV2╚
&decoder/conv2d_2/Conv2D/ReadVariableOpReadVariableOp/decoder_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02(
&decoder/conv2d_2/Conv2D/ReadVariableOpЭ
decoder/conv2d_2/Conv2DConv2D&decoder/tf.__operators__.add/AddV2:z:0.decoder/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ@*
paddingSAME*
strides
2
decoder/conv2d_2/Conv2D┐
'decoder/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp0decoder_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'decoder/conv2d_2/BiasAdd/ReadVariableOp╬
decoder/conv2d_2/BiasAddBiasAdd decoder/conv2d_2/Conv2D:output:0/decoder/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ@2
decoder/conv2d_2/BiasAddЋ
decoder/conv2d_2/ReluRelu!decoder/conv2d_2/BiasAdd:output:0*
T0*1
_output_shapes
:         ЯЯ@2
decoder/conv2d_2/Relu╚
&decoder/conv2d_3/Conv2D/ReadVariableOpReadVariableOp/decoder_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02(
&decoder/conv2d_3/Conv2D/ReadVariableOpш
decoder/conv2d_3/Conv2DConv2D#decoder/conv2d_2/Relu:activations:0.decoder/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ@*
paddingSAME*
strides
2
decoder/conv2d_3/Conv2D┐
'decoder/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp0decoder_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'decoder/conv2d_3/BiasAdd/ReadVariableOp╬
decoder/conv2d_3/BiasAddBiasAdd decoder/conv2d_3/Conv2D:output:0/decoder/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ@2
decoder/conv2d_3/BiasAddЋ
decoder/conv2d_3/ReluRelu!decoder/conv2d_3/BiasAdd:output:0*
T0*1
_output_shapes
:         ЯЯ@2
decoder/conv2d_3/Relu█
$decoder/tf.__operators__.add_1/AddV2AddV2#decoder/conv2d_3/Relu:activations:0#decoder/conv2d_2/Relu:activations:0*
T0*1
_output_shapes
:         ЯЯ@2&
$decoder/tf.__operators__.add_1/AddV2╚
&decoder/conv2d_4/Conv2D/ReadVariableOpReadVariableOp/decoder_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02(
&decoder/conv2d_4/Conv2D/ReadVariableOpЩ
decoder/conv2d_4/Conv2DConv2D(decoder/tf.__operators__.add_1/AddV2:z:0.decoder/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ*
paddingSAME*
strides
2
decoder/conv2d_4/Conv2D┐
'decoder/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp0decoder_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'decoder/conv2d_4/BiasAdd/ReadVariableOp╬
decoder/conv2d_4/BiasAddBiasAdd decoder/conv2d_4/Conv2D:output:0/decoder/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ2
decoder/conv2d_4/BiasAddЕ
0decoder/tf.clip_by_value/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?22
0decoder/tf.clip_by_value/clip_by_value/Minimum/yЁ
.decoder/tf.clip_by_value/clip_by_value/MinimumMinimum!decoder/conv2d_4/BiasAdd:output:09decoder/tf.clip_by_value/clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:         ЯЯ20
.decoder/tf.clip_by_value/clip_by_value/MinimumЎ
(decoder/tf.clip_by_value/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(decoder/tf.clip_by_value/clip_by_value/y■
&decoder/tf.clip_by_value/clip_by_valueMaximum2decoder/tf.clip_by_value/clip_by_value/Minimum:z:01decoder/tf.clip_by_value/clip_by_value/y:output:0*
T0*1
_output_shapes
:         ЯЯ2(
&decoder/tf.clip_by_value/clip_by_valueЈ
IdentityIdentity*decoder/tf.clip_by_value/clip_by_value:z:0^NoOp*
T0*1
_output_shapes
:         ЯЯ2

IdentityЂ
NoOpNoOp(^decoder/conv2d_1/BiasAdd/ReadVariableOp'^decoder/conv2d_1/Conv2D/ReadVariableOp(^decoder/conv2d_2/BiasAdd/ReadVariableOp'^decoder/conv2d_2/Conv2D/ReadVariableOp(^decoder/conv2d_3/BiasAdd/ReadVariableOp'^decoder/conv2d_3/Conv2D/ReadVariableOp(^decoder/conv2d_4/BiasAdd/ReadVariableOp'^decoder/conv2d_4/Conv2D/ReadVariableOp-^decoder/decoder_input/BiasAdd/ReadVariableOp6^decoder/decoder_input/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:           
: : : : : : : : : : 2R
'decoder/conv2d_1/BiasAdd/ReadVariableOp'decoder/conv2d_1/BiasAdd/ReadVariableOp2P
&decoder/conv2d_1/Conv2D/ReadVariableOp&decoder/conv2d_1/Conv2D/ReadVariableOp2R
'decoder/conv2d_2/BiasAdd/ReadVariableOp'decoder/conv2d_2/BiasAdd/ReadVariableOp2P
&decoder/conv2d_2/Conv2D/ReadVariableOp&decoder/conv2d_2/Conv2D/ReadVariableOp2R
'decoder/conv2d_3/BiasAdd/ReadVariableOp'decoder/conv2d_3/BiasAdd/ReadVariableOp2P
&decoder/conv2d_3/Conv2D/ReadVariableOp&decoder/conv2d_3/Conv2D/ReadVariableOp2R
'decoder/conv2d_4/BiasAdd/ReadVariableOp'decoder/conv2d_4/BiasAdd/ReadVariableOp2P
&decoder/conv2d_4/Conv2D/ReadVariableOp&decoder/conv2d_4/Conv2D/ReadVariableOp2\
,decoder/decoder_input/BiasAdd/ReadVariableOp,decoder/decoder_input/BiasAdd/ReadVariableOp2n
5decoder/decoder_input/conv2d_transpose/ReadVariableOp5decoder/decoder_input/conv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:           

 
_user_specified_nameinputs
ќ(
Ю
D__inference_decoder_layer_call_and_return_conditional_losses_5767052

inputs/
decoder_input_5766973:		@
#
decoder_input_5766975:@*
conv2d_1_5766990:@@
conv2d_1_5766992:@*
conv2d_2_5767008:@@
conv2d_2_5767010:@*
conv2d_3_5767025:@@
conv2d_3_5767027:@*
conv2d_4_5767042:@
conv2d_4_5767044:
identityѕб conv2d_1/StatefulPartitionedCallб conv2d_2/StatefulPartitionedCallб conv2d_3/StatefulPartitionedCallб conv2d_4/StatefulPartitionedCallб%decoder_input/StatefulPartitionedCall┐
%decoder_input/StatefulPartitionedCallStatefulPartitionedCallinputsdecoder_input_5766973decoder_input_5766975*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЯЯ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__inference_decoder_input_layer_call_and_return_conditional_losses_57669722'
%decoder_input/StatefulPartitionedCall╬
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall.decoder_input/StatefulPartitionedCall:output:0conv2d_1_5766990conv2d_1_5766992*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЯЯ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_57669892"
 conv2d_1/StatefulPartitionedCallп
tf.__operators__.add/AddV2AddV2)conv2d_1/StatefulPartitionedCall:output:0.decoder_input/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:         ЯЯ@2
tf.__operators__.add/AddV2Й
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCalltf.__operators__.add/AddV2:z:0conv2d_2_5767008conv2d_2_5767010*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЯЯ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_57670072"
 conv2d_2/StatefulPartitionedCall╔
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_5767025conv2d_3_5767027*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЯЯ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_57670242"
 conv2d_3/StatefulPartitionedCallО
tf.__operators__.add_1/AddV2AddV2)conv2d_3/StatefulPartitionedCall:output:0)conv2d_2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:         ЯЯ@2
tf.__operators__.add_1/AddV2└
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_1/AddV2:z:0conv2d_4_5767042conv2d_4_5767044*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЯЯ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_57670412"
 conv2d_4/StatefulPartitionedCallЎ
(tf.clip_by_value/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2*
(tf.clip_by_value/clip_by_value/Minimum/yш
&tf.clip_by_value/clip_by_value/MinimumMinimum)conv2d_4/StatefulPartitionedCall:output:01tf.clip_by_value/clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:         ЯЯ2(
&tf.clip_by_value/clip_by_value/MinimumЅ
 tf.clip_by_value/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 tf.clip_by_value/clip_by_value/yя
tf.clip_by_value/clip_by_valueMaximum*tf.clip_by_value/clip_by_value/Minimum:z:0)tf.clip_by_value/clip_by_value/y:output:0*
T0*1
_output_shapes
:         ЯЯ2 
tf.clip_by_value/clip_by_valueЄ
IdentityIdentity"tf.clip_by_value/clip_by_value:z:0^NoOp*
T0*1
_output_shapes
:         ЯЯ2

Identityѓ
NoOpNoOp!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall&^decoder_input/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:           
: : : : : : : : : : 2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2N
%decoder_input/StatefulPartitionedCall%decoder_input/StatefulPartitionedCall:W S
/
_output_shapes
:           

 
_user_specified_nameinputs
»
ц
/__inference_decoder_input_layer_call_fn_5767937

inputs!
unknown:		@

	unknown_0:@
identityѕбStatefulPartitionedCallЅ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЯЯ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__inference_decoder_input_layer_call_and_return_conditional_losses_57669722
StatefulPartitionedCallЁ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ЯЯ@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:           
: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:           

 
_user_specified_nameinputs
Ў(
ъ
D__inference_decoder_layer_call_and_return_conditional_losses_5767269
input_2/
decoder_input_5767237:		@
#
decoder_input_5767239:@*
conv2d_1_5767242:@@
conv2d_1_5767244:@*
conv2d_2_5767248:@@
conv2d_2_5767250:@*
conv2d_3_5767253:@@
conv2d_3_5767255:@*
conv2d_4_5767259:@
conv2d_4_5767261:
identityѕб conv2d_1/StatefulPartitionedCallб conv2d_2/StatefulPartitionedCallб conv2d_3/StatefulPartitionedCallб conv2d_4/StatefulPartitionedCallб%decoder_input/StatefulPartitionedCall└
%decoder_input/StatefulPartitionedCallStatefulPartitionedCallinput_2decoder_input_5767237decoder_input_5767239*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЯЯ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__inference_decoder_input_layer_call_and_return_conditional_losses_57669722'
%decoder_input/StatefulPartitionedCall╬
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall.decoder_input/StatefulPartitionedCall:output:0conv2d_1_5767242conv2d_1_5767244*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЯЯ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_57669892"
 conv2d_1/StatefulPartitionedCallп
tf.__operators__.add/AddV2AddV2)conv2d_1/StatefulPartitionedCall:output:0.decoder_input/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:         ЯЯ@2
tf.__operators__.add/AddV2Й
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCalltf.__operators__.add/AddV2:z:0conv2d_2_5767248conv2d_2_5767250*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЯЯ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_57670072"
 conv2d_2/StatefulPartitionedCall╔
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_5767253conv2d_3_5767255*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЯЯ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_57670242"
 conv2d_3/StatefulPartitionedCallО
tf.__operators__.add_1/AddV2AddV2)conv2d_3/StatefulPartitionedCall:output:0)conv2d_2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:         ЯЯ@2
tf.__operators__.add_1/AddV2└
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_1/AddV2:z:0conv2d_4_5767259conv2d_4_5767261*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЯЯ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_57670412"
 conv2d_4/StatefulPartitionedCallЎ
(tf.clip_by_value/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2*
(tf.clip_by_value/clip_by_value/Minimum/yш
&tf.clip_by_value/clip_by_value/MinimumMinimum)conv2d_4/StatefulPartitionedCall:output:01tf.clip_by_value/clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:         ЯЯ2(
&tf.clip_by_value/clip_by_value/MinimumЅ
 tf.clip_by_value/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 tf.clip_by_value/clip_by_value/yя
tf.clip_by_value/clip_by_valueMaximum*tf.clip_by_value/clip_by_value/Minimum:z:0)tf.clip_by_value/clip_by_value/y:output:0*
T0*1
_output_shapes
:         ЯЯ2 
tf.clip_by_value/clip_by_valueЄ
IdentityIdentity"tf.clip_by_value/clip_by_value:z:0^NoOp*
T0*1
_output_shapes
:         ЯЯ2

Identityѓ
NoOpNoOp!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall&^decoder_input/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:           
: : : : : : : : : : 2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2N
%decoder_input/StatefulPartitionedCall%decoder_input/StatefulPartitionedCall:X T
/
_output_shapes
:           

!
_user_specified_name	input_2
с
Э
E__inference_decoder1_layer_call_and_return_conditional_losses_5767408

inputs)
decoder_5767386:		@

decoder_5767388:@)
decoder_5767390:@@
decoder_5767392:@)
decoder_5767394:@@
decoder_5767396:@)
decoder_5767398:@@
decoder_5767400:@)
decoder_5767402:@
decoder_5767404:
identityѕбdecoder/StatefulPartitionedCall╣
decoder/StatefulPartitionedCallStatefulPartitionedCallinputsdecoder_5767386decoder_5767388decoder_5767390decoder_5767392decoder_5767394decoder_5767396decoder_5767398decoder_5767400decoder_5767402decoder_5767404*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЯЯ*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *M
fHRF
D__inference_decoder_layer_call_and_return_conditional_losses_57671862!
decoder/StatefulPartitionedCallЇ
IdentityIdentity(decoder/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ЯЯ2

Identityp
NoOpNoOp ^decoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:           
: : : : : : : : : : 2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall:W S
/
_output_shapes
:           

 
_user_specified_nameinputs
з
ц
/__inference_decoder_input_layer_call_fn_5767928

inputs!
unknown:		@

	unknown_0:@
identityѕбStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__inference_decoder_input_layer_call_and_return_conditional_losses_57668922
StatefulPartitionedCallЋ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           @2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           
: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           

 
_user_specified_nameinputs
§P
┴
D__inference_decoder_layer_call_and_return_conditional_losses_5767754

inputsP
6decoder_input_conv2d_transpose_readvariableop_resource:		@
;
-decoder_input_biasadd_readvariableop_resource:@A
'conv2d_1_conv2d_readvariableop_resource:@@6
(conv2d_1_biasadd_readvariableop_resource:@A
'conv2d_2_conv2d_readvariableop_resource:@@6
(conv2d_2_biasadd_readvariableop_resource:@A
'conv2d_3_conv2d_readvariableop_resource:@@6
(conv2d_3_biasadd_readvariableop_resource:@A
'conv2d_4_conv2d_readvariableop_resource:@6
(conv2d_4_biasadd_readvariableop_resource:
identityѕбconv2d_1/BiasAdd/ReadVariableOpбconv2d_1/Conv2D/ReadVariableOpбconv2d_2/BiasAdd/ReadVariableOpбconv2d_2/Conv2D/ReadVariableOpбconv2d_3/BiasAdd/ReadVariableOpбconv2d_3/Conv2D/ReadVariableOpбconv2d_4/BiasAdd/ReadVariableOpбconv2d_4/Conv2D/ReadVariableOpб$decoder_input/BiasAdd/ReadVariableOpб-decoder_input/conv2d_transpose/ReadVariableOp`
decoder_input/ShapeShapeinputs*
T0*
_output_shapes
:2
decoder_input/Shapeљ
!decoder_input/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!decoder_input/strided_slice/stackћ
#decoder_input/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#decoder_input/strided_slice/stack_1ћ
#decoder_input/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#decoder_input/strided_slice/stack_2Х
decoder_input/strided_sliceStridedSlicedecoder_input/Shape:output:0*decoder_input/strided_slice/stack:output:0,decoder_input/strided_slice/stack_1:output:0,decoder_input/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
decoder_input/strided_sliceq
decoder_input/stack/1Const*
_output_shapes
: *
dtype0*
value
B :Я2
decoder_input/stack/1q
decoder_input/stack/2Const*
_output_shapes
: *
dtype0*
value
B :Я2
decoder_input/stack/2p
decoder_input/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
decoder_input/stack/3Т
decoder_input/stackPack$decoder_input/strided_slice:output:0decoder_input/stack/1:output:0decoder_input/stack/2:output:0decoder_input/stack/3:output:0*
N*
T0*
_output_shapes
:2
decoder_input/stackћ
#decoder_input/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#decoder_input/strided_slice_1/stackў
%decoder_input/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%decoder_input/strided_slice_1/stack_1ў
%decoder_input/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%decoder_input/strided_slice_1/stack_2└
decoder_input/strided_slice_1StridedSlicedecoder_input/stack:output:0,decoder_input/strided_slice_1/stack:output:0.decoder_input/strided_slice_1/stack_1:output:0.decoder_input/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
decoder_input/strided_slice_1П
-decoder_input/conv2d_transpose/ReadVariableOpReadVariableOp6decoder_input_conv2d_transpose_readvariableop_resource*&
_output_shapes
:		@
*
dtype02/
-decoder_input/conv2d_transpose/ReadVariableOpў
decoder_input/conv2d_transposeConv2DBackpropInputdecoder_input/stack:output:05decoder_input/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*1
_output_shapes
:         ЯЯ@*
paddingSAME*
strides
2 
decoder_input/conv2d_transposeХ
$decoder_input/BiasAdd/ReadVariableOpReadVariableOp-decoder_input_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02&
$decoder_input/BiasAdd/ReadVariableOp╠
decoder_input/BiasAddBiasAdd'decoder_input/conv2d_transpose:output:0,decoder_input/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ@2
decoder_input/BiasAddї
decoder_input/ReluReludecoder_input/BiasAdd:output:0*
T0*1
_output_shapes
:         ЯЯ@2
decoder_input/Relu░
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_1/Conv2D/ReadVariableOp┌
conv2d_1/Conv2DConv2D decoder_input/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ@*
paddingSAME*
strides
2
conv2d_1/Conv2DД
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp«
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ@2
conv2d_1/BiasAdd}
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*1
_output_shapes
:         ЯЯ@2
conv2d_1/Relu╝
tf.__operators__.add/AddV2AddV2conv2d_1/Relu:activations:0 decoder_input/Relu:activations:0*
T0*1
_output_shapes
:         ЯЯ@2
tf.__operators__.add/AddV2░
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_2/Conv2D/ReadVariableOpп
conv2d_2/Conv2DConv2Dtf.__operators__.add/AddV2:z:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ@*
paddingSAME*
strides
2
conv2d_2/Conv2DД
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp«
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ@2
conv2d_2/BiasAdd}
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*1
_output_shapes
:         ЯЯ@2
conv2d_2/Relu░
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_3/Conv2D/ReadVariableOpН
conv2d_3/Conv2DConv2Dconv2d_2/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ@*
paddingSAME*
strides
2
conv2d_3/Conv2DД
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp«
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ@2
conv2d_3/BiasAdd}
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*1
_output_shapes
:         ЯЯ@2
conv2d_3/Relu╗
tf.__operators__.add_1/AddV2AddV2conv2d_3/Relu:activations:0conv2d_2/Relu:activations:0*
T0*1
_output_shapes
:         ЯЯ@2
tf.__operators__.add_1/AddV2░
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02 
conv2d_4/Conv2D/ReadVariableOp┌
conv2d_4/Conv2DConv2D tf.__operators__.add_1/AddV2:z:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ*
paddingSAME*
strides
2
conv2d_4/Conv2DД
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp«
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ2
conv2d_4/BiasAddЎ
(tf.clip_by_value/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2*
(tf.clip_by_value/clip_by_value/Minimum/yт
&tf.clip_by_value/clip_by_value/MinimumMinimumconv2d_4/BiasAdd:output:01tf.clip_by_value/clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:         ЯЯ2(
&tf.clip_by_value/clip_by_value/MinimumЅ
 tf.clip_by_value/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 tf.clip_by_value/clip_by_value/yя
tf.clip_by_value/clip_by_valueMaximum*tf.clip_by_value/clip_by_value/Minimum:z:0)tf.clip_by_value/clip_by_value/y:output:0*
T0*1
_output_shapes
:         ЯЯ2 
tf.clip_by_value/clip_by_valueЄ
IdentityIdentity"tf.clip_by_value/clip_by_value:z:0^NoOp*
T0*1
_output_shapes
:         ЯЯ2

Identity▒
NoOpNoOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp%^decoder_input/BiasAdd/ReadVariableOp.^decoder_input/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:           
: : : : : : : : : : 2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2L
$decoder_input/BiasAdd/ReadVariableOp$decoder_input/BiasAdd/ReadVariableOp2^
-decoder_input/conv2d_transpose/ReadVariableOp-decoder_input/conv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:           

 
_user_specified_nameinputs
э
■
E__inference_conv2d_2_layer_call_and_return_conditional_losses_5767968

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpЦ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ@*
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpі
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ@2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:         ЯЯ@2
Reluw
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:         ЯЯ@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ЯЯ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:         ЯЯ@
 
_user_specified_nameinputs
Е
Ъ
*__inference_conv2d_4_layer_call_fn_5768016

inputs!
unknown:@
	unknown_0:
identityѕбStatefulPartitionedCallё
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЯЯ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_57670412
StatefulPartitionedCallЁ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ЯЯ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ЯЯ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ЯЯ@
 
_user_specified_nameinputs
А
Ќ
J__inference_decoder_input_layer_call_and_return_conditional_losses_5766972

inputsB
(conv2d_transpose_readvariableop_resource:		@
-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбconv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceU
stack/1Const*
_output_shapes
: *
dtype0*
value
B :Я2	
stack/1U
stack/2Const*
_output_shapes
: *
dtype0*
value
B :Я2	
stack/2T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/3њ
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2В
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1│
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:		@
*
dtype02!
conv2d_transpose/ReadVariableOpЯ
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*1
_output_shapes
:         ЯЯ@*
paddingSAME*
strides
2
conv2d_transposeї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpћ
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ@2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:         ЯЯ@2
Reluw
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:         ЯЯ@2

IdentityЅ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:           
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:           

 
_user_specified_nameinputs
Е
Ъ
*__inference_conv2d_3_layer_call_fn_5767997

inputs!
unknown:@@
	unknown_0:@
identityѕбStatefulPartitionedCallё
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЯЯ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_57670242
StatefulPartitionedCallЁ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ЯЯ@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ЯЯ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ЯЯ@
 
_user_specified_nameinputs
Њ
ќ
)__inference_decoder_layer_call_fn_5767836

inputs!
unknown:		@

	unknown_0:@#
	unknown_1:@@
	unknown_2:@#
	unknown_3:@@
	unknown_4:@#
	unknown_5:@@
	unknown_6:@#
	unknown_7:@
	unknown_8:
identityѕбStatefulPartitionedCallв
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЯЯ*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *M
fHRF
D__inference_decoder_layer_call_and_return_conditional_losses_57670522
StatefulPartitionedCallЁ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ЯЯ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:           
: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:           

 
_user_specified_nameinputs
ѓ
Ў
%__inference_signature_wrapper_5767533
decoder_input!
unknown:		@

	unknown_0:@#
	unknown_1:@@
	unknown_2:@#
	unknown_3:@@
	unknown_4:@#
	unknown_5:@@
	unknown_6:@#
	unknown_7:@
	unknown_8:
identityѕбStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCalldecoder_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЯЯ*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *+
f&R$
"__inference__wrapped_model_57668542
StatefulPartitionedCallЁ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ЯЯ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:           
: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
/
_output_shapes
:           

'
_user_specified_namedecoder_input
«j
Щ

"__inference__wrapped_model_5766854
decoder_inputa
Gdecoder1_decoder_decoder_input_conv2d_transpose_readvariableop_resource:		@
L
>decoder1_decoder_decoder_input_biasadd_readvariableop_resource:@R
8decoder1_decoder_conv2d_1_conv2d_readvariableop_resource:@@G
9decoder1_decoder_conv2d_1_biasadd_readvariableop_resource:@R
8decoder1_decoder_conv2d_2_conv2d_readvariableop_resource:@@G
9decoder1_decoder_conv2d_2_biasadd_readvariableop_resource:@R
8decoder1_decoder_conv2d_3_conv2d_readvariableop_resource:@@G
9decoder1_decoder_conv2d_3_biasadd_readvariableop_resource:@R
8decoder1_decoder_conv2d_4_conv2d_readvariableop_resource:@G
9decoder1_decoder_conv2d_4_biasadd_readvariableop_resource:
identityѕб0decoder1/decoder/conv2d_1/BiasAdd/ReadVariableOpб/decoder1/decoder/conv2d_1/Conv2D/ReadVariableOpб0decoder1/decoder/conv2d_2/BiasAdd/ReadVariableOpб/decoder1/decoder/conv2d_2/Conv2D/ReadVariableOpб0decoder1/decoder/conv2d_3/BiasAdd/ReadVariableOpб/decoder1/decoder/conv2d_3/Conv2D/ReadVariableOpб0decoder1/decoder/conv2d_4/BiasAdd/ReadVariableOpб/decoder1/decoder/conv2d_4/Conv2D/ReadVariableOpб5decoder1/decoder/decoder_input/BiasAdd/ReadVariableOpб>decoder1/decoder/decoder_input/conv2d_transpose/ReadVariableOpЅ
$decoder1/decoder/decoder_input/ShapeShapedecoder_input*
T0*
_output_shapes
:2&
$decoder1/decoder/decoder_input/Shape▓
2decoder1/decoder/decoder_input/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2decoder1/decoder/decoder_input/strided_slice/stackХ
4decoder1/decoder/decoder_input/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4decoder1/decoder/decoder_input/strided_slice/stack_1Х
4decoder1/decoder/decoder_input/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4decoder1/decoder/decoder_input/strided_slice/stack_2ю
,decoder1/decoder/decoder_input/strided_sliceStridedSlice-decoder1/decoder/decoder_input/Shape:output:0;decoder1/decoder/decoder_input/strided_slice/stack:output:0=decoder1/decoder/decoder_input/strided_slice/stack_1:output:0=decoder1/decoder/decoder_input/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,decoder1/decoder/decoder_input/strided_sliceЊ
&decoder1/decoder/decoder_input/stack/1Const*
_output_shapes
: *
dtype0*
value
B :Я2(
&decoder1/decoder/decoder_input/stack/1Њ
&decoder1/decoder/decoder_input/stack/2Const*
_output_shapes
: *
dtype0*
value
B :Я2(
&decoder1/decoder/decoder_input/stack/2њ
&decoder1/decoder/decoder_input/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2(
&decoder1/decoder/decoder_input/stack/3╠
$decoder1/decoder/decoder_input/stackPack5decoder1/decoder/decoder_input/strided_slice:output:0/decoder1/decoder/decoder_input/stack/1:output:0/decoder1/decoder/decoder_input/stack/2:output:0/decoder1/decoder/decoder_input/stack/3:output:0*
N*
T0*
_output_shapes
:2&
$decoder1/decoder/decoder_input/stackХ
4decoder1/decoder/decoder_input/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4decoder1/decoder/decoder_input/strided_slice_1/stack║
6decoder1/decoder/decoder_input/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6decoder1/decoder/decoder_input/strided_slice_1/stack_1║
6decoder1/decoder/decoder_input/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6decoder1/decoder/decoder_input/strided_slice_1/stack_2д
.decoder1/decoder/decoder_input/strided_slice_1StridedSlice-decoder1/decoder/decoder_input/stack:output:0=decoder1/decoder/decoder_input/strided_slice_1/stack:output:0?decoder1/decoder/decoder_input/strided_slice_1/stack_1:output:0?decoder1/decoder/decoder_input/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.decoder1/decoder/decoder_input/strided_slice_1љ
>decoder1/decoder/decoder_input/conv2d_transpose/ReadVariableOpReadVariableOpGdecoder1_decoder_decoder_input_conv2d_transpose_readvariableop_resource*&
_output_shapes
:		@
*
dtype02@
>decoder1/decoder/decoder_input/conv2d_transpose/ReadVariableOpс
/decoder1/decoder/decoder_input/conv2d_transposeConv2DBackpropInput-decoder1/decoder/decoder_input/stack:output:0Fdecoder1/decoder/decoder_input/conv2d_transpose/ReadVariableOp:value:0decoder_input*
T0*1
_output_shapes
:         ЯЯ@*
paddingSAME*
strides
21
/decoder1/decoder/decoder_input/conv2d_transposeж
5decoder1/decoder/decoder_input/BiasAdd/ReadVariableOpReadVariableOp>decoder1_decoder_decoder_input_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype027
5decoder1/decoder/decoder_input/BiasAdd/ReadVariableOpљ
&decoder1/decoder/decoder_input/BiasAddBiasAdd8decoder1/decoder/decoder_input/conv2d_transpose:output:0=decoder1/decoder/decoder_input/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ@2(
&decoder1/decoder/decoder_input/BiasAdd┐
#decoder1/decoder/decoder_input/ReluRelu/decoder1/decoder/decoder_input/BiasAdd:output:0*
T0*1
_output_shapes
:         ЯЯ@2%
#decoder1/decoder/decoder_input/Reluс
/decoder1/decoder/conv2d_1/Conv2D/ReadVariableOpReadVariableOp8decoder1_decoder_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype021
/decoder1/decoder/conv2d_1/Conv2D/ReadVariableOpъ
 decoder1/decoder/conv2d_1/Conv2DConv2D1decoder1/decoder/decoder_input/Relu:activations:07decoder1/decoder/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ@*
paddingSAME*
strides
2"
 decoder1/decoder/conv2d_1/Conv2D┌
0decoder1/decoder/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp9decoder1_decoder_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0decoder1/decoder/conv2d_1/BiasAdd/ReadVariableOpЫ
!decoder1/decoder/conv2d_1/BiasAddBiasAdd)decoder1/decoder/conv2d_1/Conv2D:output:08decoder1/decoder/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ@2#
!decoder1/decoder/conv2d_1/BiasAdd░
decoder1/decoder/conv2d_1/ReluRelu*decoder1/decoder/conv2d_1/BiasAdd:output:0*
T0*1
_output_shapes
:         ЯЯ@2 
decoder1/decoder/conv2d_1/Reluђ
+decoder1/decoder/tf.__operators__.add/AddV2AddV2,decoder1/decoder/conv2d_1/Relu:activations:01decoder1/decoder/decoder_input/Relu:activations:0*
T0*1
_output_shapes
:         ЯЯ@2-
+decoder1/decoder/tf.__operators__.add/AddV2с
/decoder1/decoder/conv2d_2/Conv2D/ReadVariableOpReadVariableOp8decoder1_decoder_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype021
/decoder1/decoder/conv2d_2/Conv2D/ReadVariableOpю
 decoder1/decoder/conv2d_2/Conv2DConv2D/decoder1/decoder/tf.__operators__.add/AddV2:z:07decoder1/decoder/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ@*
paddingSAME*
strides
2"
 decoder1/decoder/conv2d_2/Conv2D┌
0decoder1/decoder/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp9decoder1_decoder_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0decoder1/decoder/conv2d_2/BiasAdd/ReadVariableOpЫ
!decoder1/decoder/conv2d_2/BiasAddBiasAdd)decoder1/decoder/conv2d_2/Conv2D:output:08decoder1/decoder/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ@2#
!decoder1/decoder/conv2d_2/BiasAdd░
decoder1/decoder/conv2d_2/ReluRelu*decoder1/decoder/conv2d_2/BiasAdd:output:0*
T0*1
_output_shapes
:         ЯЯ@2 
decoder1/decoder/conv2d_2/Reluс
/decoder1/decoder/conv2d_3/Conv2D/ReadVariableOpReadVariableOp8decoder1_decoder_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype021
/decoder1/decoder/conv2d_3/Conv2D/ReadVariableOpЎ
 decoder1/decoder/conv2d_3/Conv2DConv2D,decoder1/decoder/conv2d_2/Relu:activations:07decoder1/decoder/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ@*
paddingSAME*
strides
2"
 decoder1/decoder/conv2d_3/Conv2D┌
0decoder1/decoder/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp9decoder1_decoder_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0decoder1/decoder/conv2d_3/BiasAdd/ReadVariableOpЫ
!decoder1/decoder/conv2d_3/BiasAddBiasAdd)decoder1/decoder/conv2d_3/Conv2D:output:08decoder1/decoder/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ@2#
!decoder1/decoder/conv2d_3/BiasAdd░
decoder1/decoder/conv2d_3/ReluRelu*decoder1/decoder/conv2d_3/BiasAdd:output:0*
T0*1
_output_shapes
:         ЯЯ@2 
decoder1/decoder/conv2d_3/Relu 
-decoder1/decoder/tf.__operators__.add_1/AddV2AddV2,decoder1/decoder/conv2d_3/Relu:activations:0,decoder1/decoder/conv2d_2/Relu:activations:0*
T0*1
_output_shapes
:         ЯЯ@2/
-decoder1/decoder/tf.__operators__.add_1/AddV2с
/decoder1/decoder/conv2d_4/Conv2D/ReadVariableOpReadVariableOp8decoder1_decoder_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype021
/decoder1/decoder/conv2d_4/Conv2D/ReadVariableOpъ
 decoder1/decoder/conv2d_4/Conv2DConv2D1decoder1/decoder/tf.__operators__.add_1/AddV2:z:07decoder1/decoder/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ*
paddingSAME*
strides
2"
 decoder1/decoder/conv2d_4/Conv2D┌
0decoder1/decoder/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp9decoder1_decoder_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0decoder1/decoder/conv2d_4/BiasAdd/ReadVariableOpЫ
!decoder1/decoder/conv2d_4/BiasAddBiasAdd)decoder1/decoder/conv2d_4/Conv2D:output:08decoder1/decoder/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ2#
!decoder1/decoder/conv2d_4/BiasAdd╗
9decoder1/decoder/tf.clip_by_value/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2;
9decoder1/decoder/tf.clip_by_value/clip_by_value/Minimum/yЕ
7decoder1/decoder/tf.clip_by_value/clip_by_value/MinimumMinimum*decoder1/decoder/conv2d_4/BiasAdd:output:0Bdecoder1/decoder/tf.clip_by_value/clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:         ЯЯ29
7decoder1/decoder/tf.clip_by_value/clip_by_value/MinimumФ
1decoder1/decoder/tf.clip_by_value/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    23
1decoder1/decoder/tf.clip_by_value/clip_by_value/yб
/decoder1/decoder/tf.clip_by_value/clip_by_valueMaximum;decoder1/decoder/tf.clip_by_value/clip_by_value/Minimum:z:0:decoder1/decoder/tf.clip_by_value/clip_by_value/y:output:0*
T0*1
_output_shapes
:         ЯЯ21
/decoder1/decoder/tf.clip_by_value/clip_by_valueў
IdentityIdentity3decoder1/decoder/tf.clip_by_value/clip_by_value:z:0^NoOp*
T0*1
_output_shapes
:         ЯЯ2

Identity█
NoOpNoOp1^decoder1/decoder/conv2d_1/BiasAdd/ReadVariableOp0^decoder1/decoder/conv2d_1/Conv2D/ReadVariableOp1^decoder1/decoder/conv2d_2/BiasAdd/ReadVariableOp0^decoder1/decoder/conv2d_2/Conv2D/ReadVariableOp1^decoder1/decoder/conv2d_3/BiasAdd/ReadVariableOp0^decoder1/decoder/conv2d_3/Conv2D/ReadVariableOp1^decoder1/decoder/conv2d_4/BiasAdd/ReadVariableOp0^decoder1/decoder/conv2d_4/Conv2D/ReadVariableOp6^decoder1/decoder/decoder_input/BiasAdd/ReadVariableOp?^decoder1/decoder/decoder_input/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:           
: : : : : : : : : : 2d
0decoder1/decoder/conv2d_1/BiasAdd/ReadVariableOp0decoder1/decoder/conv2d_1/BiasAdd/ReadVariableOp2b
/decoder1/decoder/conv2d_1/Conv2D/ReadVariableOp/decoder1/decoder/conv2d_1/Conv2D/ReadVariableOp2d
0decoder1/decoder/conv2d_2/BiasAdd/ReadVariableOp0decoder1/decoder/conv2d_2/BiasAdd/ReadVariableOp2b
/decoder1/decoder/conv2d_2/Conv2D/ReadVariableOp/decoder1/decoder/conv2d_2/Conv2D/ReadVariableOp2d
0decoder1/decoder/conv2d_3/BiasAdd/ReadVariableOp0decoder1/decoder/conv2d_3/BiasAdd/ReadVariableOp2b
/decoder1/decoder/conv2d_3/Conv2D/ReadVariableOp/decoder1/decoder/conv2d_3/Conv2D/ReadVariableOp2d
0decoder1/decoder/conv2d_4/BiasAdd/ReadVariableOp0decoder1/decoder/conv2d_4/BiasAdd/ReadVariableOp2b
/decoder1/decoder/conv2d_4/Conv2D/ReadVariableOp/decoder1/decoder/conv2d_4/Conv2D/ReadVariableOp2n
5decoder1/decoder/decoder_input/BiasAdd/ReadVariableOp5decoder1/decoder/decoder_input/BiasAdd/ReadVariableOp2ђ
>decoder1/decoder/decoder_input/conv2d_transpose/ReadVariableOp>decoder1/decoder/decoder_input/conv2d_transpose/ReadVariableOp:^ Z
/
_output_shapes
:           

'
_user_specified_namedecoder_input
э
■
E__inference_conv2d_3_layer_call_and_return_conditional_losses_5767024

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpЦ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ@*
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpі
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ@2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:         ЯЯ@2
Reluw
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:         ЯЯ@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ЯЯ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:         ЯЯ@
 
_user_specified_nameinputs
Ў(
ъ
D__inference_decoder_layer_call_and_return_conditional_losses_5767304
input_2/
decoder_input_5767272:		@
#
decoder_input_5767274:@*
conv2d_1_5767277:@@
conv2d_1_5767279:@*
conv2d_2_5767283:@@
conv2d_2_5767285:@*
conv2d_3_5767288:@@
conv2d_3_5767290:@*
conv2d_4_5767294:@
conv2d_4_5767296:
identityѕб conv2d_1/StatefulPartitionedCallб conv2d_2/StatefulPartitionedCallб conv2d_3/StatefulPartitionedCallб conv2d_4/StatefulPartitionedCallб%decoder_input/StatefulPartitionedCall└
%decoder_input/StatefulPartitionedCallStatefulPartitionedCallinput_2decoder_input_5767272decoder_input_5767274*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЯЯ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__inference_decoder_input_layer_call_and_return_conditional_losses_57669722'
%decoder_input/StatefulPartitionedCall╬
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall.decoder_input/StatefulPartitionedCall:output:0conv2d_1_5767277conv2d_1_5767279*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЯЯ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_57669892"
 conv2d_1/StatefulPartitionedCallп
tf.__operators__.add/AddV2AddV2)conv2d_1/StatefulPartitionedCall:output:0.decoder_input/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:         ЯЯ@2
tf.__operators__.add/AddV2Й
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCalltf.__operators__.add/AddV2:z:0conv2d_2_5767283conv2d_2_5767285*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЯЯ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_57670072"
 conv2d_2/StatefulPartitionedCall╔
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_5767288conv2d_3_5767290*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЯЯ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_57670242"
 conv2d_3/StatefulPartitionedCallО
tf.__operators__.add_1/AddV2AddV2)conv2d_3/StatefulPartitionedCall:output:0)conv2d_2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:         ЯЯ@2
tf.__operators__.add_1/AddV2└
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_1/AddV2:z:0conv2d_4_5767294conv2d_4_5767296*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЯЯ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_57670412"
 conv2d_4/StatefulPartitionedCallЎ
(tf.clip_by_value/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2*
(tf.clip_by_value/clip_by_value/Minimum/yш
&tf.clip_by_value/clip_by_value/MinimumMinimum)conv2d_4/StatefulPartitionedCall:output:01tf.clip_by_value/clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:         ЯЯ2(
&tf.clip_by_value/clip_by_value/MinimumЅ
 tf.clip_by_value/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 tf.clip_by_value/clip_by_value/yя
tf.clip_by_value/clip_by_valueMaximum*tf.clip_by_value/clip_by_value/Minimum:z:0)tf.clip_by_value/clip_by_value/y:output:0*
T0*1
_output_shapes
:         ЯЯ2 
tf.clip_by_value/clip_by_valueЄ
IdentityIdentity"tf.clip_by_value/clip_by_value:z:0^NoOp*
T0*1
_output_shapes
:         ЯЯ2

Identityѓ
NoOpNoOp!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall&^decoder_input/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:           
: : : : : : : : : : 2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2N
%decoder_input/StatefulPartitionedCall%decoder_input/StatefulPartitionedCall:X T
/
_output_shapes
:           

!
_user_specified_name	input_2
Е
Ъ
*__inference_conv2d_1_layer_call_fn_5767957

inputs!
unknown:@@
	unknown_0:@
identityѕбStatefulPartitionedCallё
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЯЯ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_57669892
StatefulPartitionedCallЁ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ЯЯ@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ЯЯ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ЯЯ@
 
_user_specified_nameinputs
Ч\
Р	
E__inference_decoder1_layer_call_and_return_conditional_losses_5767590

inputsX
>decoder_decoder_input_conv2d_transpose_readvariableop_resource:		@
C
5decoder_decoder_input_biasadd_readvariableop_resource:@I
/decoder_conv2d_1_conv2d_readvariableop_resource:@@>
0decoder_conv2d_1_biasadd_readvariableop_resource:@I
/decoder_conv2d_2_conv2d_readvariableop_resource:@@>
0decoder_conv2d_2_biasadd_readvariableop_resource:@I
/decoder_conv2d_3_conv2d_readvariableop_resource:@@>
0decoder_conv2d_3_biasadd_readvariableop_resource:@I
/decoder_conv2d_4_conv2d_readvariableop_resource:@>
0decoder_conv2d_4_biasadd_readvariableop_resource:
identityѕб'decoder/conv2d_1/BiasAdd/ReadVariableOpб&decoder/conv2d_1/Conv2D/ReadVariableOpб'decoder/conv2d_2/BiasAdd/ReadVariableOpб&decoder/conv2d_2/Conv2D/ReadVariableOpб'decoder/conv2d_3/BiasAdd/ReadVariableOpб&decoder/conv2d_3/Conv2D/ReadVariableOpб'decoder/conv2d_4/BiasAdd/ReadVariableOpб&decoder/conv2d_4/Conv2D/ReadVariableOpб,decoder/decoder_input/BiasAdd/ReadVariableOpб5decoder/decoder_input/conv2d_transpose/ReadVariableOpp
decoder/decoder_input/ShapeShapeinputs*
T0*
_output_shapes
:2
decoder/decoder_input/Shapeа
)decoder/decoder_input/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)decoder/decoder_input/strided_slice/stackц
+decoder/decoder_input/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+decoder/decoder_input/strided_slice/stack_1ц
+decoder/decoder_input/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+decoder/decoder_input/strided_slice/stack_2Т
#decoder/decoder_input/strided_sliceStridedSlice$decoder/decoder_input/Shape:output:02decoder/decoder_input/strided_slice/stack:output:04decoder/decoder_input/strided_slice/stack_1:output:04decoder/decoder_input/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#decoder/decoder_input/strided_sliceЂ
decoder/decoder_input/stack/1Const*
_output_shapes
: *
dtype0*
value
B :Я2
decoder/decoder_input/stack/1Ђ
decoder/decoder_input/stack/2Const*
_output_shapes
: *
dtype0*
value
B :Я2
decoder/decoder_input/stack/2ђ
decoder/decoder_input/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
decoder/decoder_input/stack/3ќ
decoder/decoder_input/stackPack,decoder/decoder_input/strided_slice:output:0&decoder/decoder_input/stack/1:output:0&decoder/decoder_input/stack/2:output:0&decoder/decoder_input/stack/3:output:0*
N*
T0*
_output_shapes
:2
decoder/decoder_input/stackц
+decoder/decoder_input/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+decoder/decoder_input/strided_slice_1/stackе
-decoder/decoder_input/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-decoder/decoder_input/strided_slice_1/stack_1е
-decoder/decoder_input/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-decoder/decoder_input/strided_slice_1/stack_2­
%decoder/decoder_input/strided_slice_1StridedSlice$decoder/decoder_input/stack:output:04decoder/decoder_input/strided_slice_1/stack:output:06decoder/decoder_input/strided_slice_1/stack_1:output:06decoder/decoder_input/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%decoder/decoder_input/strided_slice_1ш
5decoder/decoder_input/conv2d_transpose/ReadVariableOpReadVariableOp>decoder_decoder_input_conv2d_transpose_readvariableop_resource*&
_output_shapes
:		@
*
dtype027
5decoder/decoder_input/conv2d_transpose/ReadVariableOpИ
&decoder/decoder_input/conv2d_transposeConv2DBackpropInput$decoder/decoder_input/stack:output:0=decoder/decoder_input/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*1
_output_shapes
:         ЯЯ@*
paddingSAME*
strides
2(
&decoder/decoder_input/conv2d_transpose╬
,decoder/decoder_input/BiasAdd/ReadVariableOpReadVariableOp5decoder_decoder_input_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,decoder/decoder_input/BiasAdd/ReadVariableOpВ
decoder/decoder_input/BiasAddBiasAdd/decoder/decoder_input/conv2d_transpose:output:04decoder/decoder_input/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ@2
decoder/decoder_input/BiasAddц
decoder/decoder_input/ReluRelu&decoder/decoder_input/BiasAdd:output:0*
T0*1
_output_shapes
:         ЯЯ@2
decoder/decoder_input/Relu╚
&decoder/conv2d_1/Conv2D/ReadVariableOpReadVariableOp/decoder_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02(
&decoder/conv2d_1/Conv2D/ReadVariableOpЩ
decoder/conv2d_1/Conv2DConv2D(decoder/decoder_input/Relu:activations:0.decoder/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ@*
paddingSAME*
strides
2
decoder/conv2d_1/Conv2D┐
'decoder/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp0decoder_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'decoder/conv2d_1/BiasAdd/ReadVariableOp╬
decoder/conv2d_1/BiasAddBiasAdd decoder/conv2d_1/Conv2D:output:0/decoder/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ@2
decoder/conv2d_1/BiasAddЋ
decoder/conv2d_1/ReluRelu!decoder/conv2d_1/BiasAdd:output:0*
T0*1
_output_shapes
:         ЯЯ@2
decoder/conv2d_1/Relu▄
"decoder/tf.__operators__.add/AddV2AddV2#decoder/conv2d_1/Relu:activations:0(decoder/decoder_input/Relu:activations:0*
T0*1
_output_shapes
:         ЯЯ@2$
"decoder/tf.__operators__.add/AddV2╚
&decoder/conv2d_2/Conv2D/ReadVariableOpReadVariableOp/decoder_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02(
&decoder/conv2d_2/Conv2D/ReadVariableOpЭ
decoder/conv2d_2/Conv2DConv2D&decoder/tf.__operators__.add/AddV2:z:0.decoder/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ@*
paddingSAME*
strides
2
decoder/conv2d_2/Conv2D┐
'decoder/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp0decoder_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'decoder/conv2d_2/BiasAdd/ReadVariableOp╬
decoder/conv2d_2/BiasAddBiasAdd decoder/conv2d_2/Conv2D:output:0/decoder/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ@2
decoder/conv2d_2/BiasAddЋ
decoder/conv2d_2/ReluRelu!decoder/conv2d_2/BiasAdd:output:0*
T0*1
_output_shapes
:         ЯЯ@2
decoder/conv2d_2/Relu╚
&decoder/conv2d_3/Conv2D/ReadVariableOpReadVariableOp/decoder_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02(
&decoder/conv2d_3/Conv2D/ReadVariableOpш
decoder/conv2d_3/Conv2DConv2D#decoder/conv2d_2/Relu:activations:0.decoder/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ@*
paddingSAME*
strides
2
decoder/conv2d_3/Conv2D┐
'decoder/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp0decoder_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'decoder/conv2d_3/BiasAdd/ReadVariableOp╬
decoder/conv2d_3/BiasAddBiasAdd decoder/conv2d_3/Conv2D:output:0/decoder/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ@2
decoder/conv2d_3/BiasAddЋ
decoder/conv2d_3/ReluRelu!decoder/conv2d_3/BiasAdd:output:0*
T0*1
_output_shapes
:         ЯЯ@2
decoder/conv2d_3/Relu█
$decoder/tf.__operators__.add_1/AddV2AddV2#decoder/conv2d_3/Relu:activations:0#decoder/conv2d_2/Relu:activations:0*
T0*1
_output_shapes
:         ЯЯ@2&
$decoder/tf.__operators__.add_1/AddV2╚
&decoder/conv2d_4/Conv2D/ReadVariableOpReadVariableOp/decoder_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02(
&decoder/conv2d_4/Conv2D/ReadVariableOpЩ
decoder/conv2d_4/Conv2DConv2D(decoder/tf.__operators__.add_1/AddV2:z:0.decoder/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ*
paddingSAME*
strides
2
decoder/conv2d_4/Conv2D┐
'decoder/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp0decoder_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'decoder/conv2d_4/BiasAdd/ReadVariableOp╬
decoder/conv2d_4/BiasAddBiasAdd decoder/conv2d_4/Conv2D:output:0/decoder/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ2
decoder/conv2d_4/BiasAddЕ
0decoder/tf.clip_by_value/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?22
0decoder/tf.clip_by_value/clip_by_value/Minimum/yЁ
.decoder/tf.clip_by_value/clip_by_value/MinimumMinimum!decoder/conv2d_4/BiasAdd:output:09decoder/tf.clip_by_value/clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:         ЯЯ20
.decoder/tf.clip_by_value/clip_by_value/MinimumЎ
(decoder/tf.clip_by_value/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(decoder/tf.clip_by_value/clip_by_value/y■
&decoder/tf.clip_by_value/clip_by_valueMaximum2decoder/tf.clip_by_value/clip_by_value/Minimum:z:01decoder/tf.clip_by_value/clip_by_value/y:output:0*
T0*1
_output_shapes
:         ЯЯ2(
&decoder/tf.clip_by_value/clip_by_valueЈ
IdentityIdentity*decoder/tf.clip_by_value/clip_by_value:z:0^NoOp*
T0*1
_output_shapes
:         ЯЯ2

IdentityЂ
NoOpNoOp(^decoder/conv2d_1/BiasAdd/ReadVariableOp'^decoder/conv2d_1/Conv2D/ReadVariableOp(^decoder/conv2d_2/BiasAdd/ReadVariableOp'^decoder/conv2d_2/Conv2D/ReadVariableOp(^decoder/conv2d_3/BiasAdd/ReadVariableOp'^decoder/conv2d_3/Conv2D/ReadVariableOp(^decoder/conv2d_4/BiasAdd/ReadVariableOp'^decoder/conv2d_4/Conv2D/ReadVariableOp-^decoder/decoder_input/BiasAdd/ReadVariableOp6^decoder/decoder_input/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:           
: : : : : : : : : : 2R
'decoder/conv2d_1/BiasAdd/ReadVariableOp'decoder/conv2d_1/BiasAdd/ReadVariableOp2P
&decoder/conv2d_1/Conv2D/ReadVariableOp&decoder/conv2d_1/Conv2D/ReadVariableOp2R
'decoder/conv2d_2/BiasAdd/ReadVariableOp'decoder/conv2d_2/BiasAdd/ReadVariableOp2P
&decoder/conv2d_2/Conv2D/ReadVariableOp&decoder/conv2d_2/Conv2D/ReadVariableOp2R
'decoder/conv2d_3/BiasAdd/ReadVariableOp'decoder/conv2d_3/BiasAdd/ReadVariableOp2P
&decoder/conv2d_3/Conv2D/ReadVariableOp&decoder/conv2d_3/Conv2D/ReadVariableOp2R
'decoder/conv2d_4/BiasAdd/ReadVariableOp'decoder/conv2d_4/BiasAdd/ReadVariableOp2P
&decoder/conv2d_4/Conv2D/ReadVariableOp&decoder/conv2d_4/Conv2D/ReadVariableOp2\
,decoder/decoder_input/BiasAdd/ReadVariableOp,decoder/decoder_input/BiasAdd/ReadVariableOp2n
5decoder/decoder_input/conv2d_transpose/ReadVariableOp5decoder/decoder_input/conv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:           

 
_user_specified_nameinputs
Љ
■
E__inference_conv2d_4_layer_call_and_return_conditional_losses_5767041

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpЦ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ*
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpі
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ2	
BiasAddu
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:         ЯЯ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ЯЯ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:         ЯЯ@
 
_user_specified_nameinputs
ќ
Ќ
)__inference_decoder_layer_call_fn_5767075
input_2!
unknown:		@

	unknown_0:@#
	unknown_1:@@
	unknown_2:@#
	unknown_3:@@
	unknown_4:@#
	unknown_5:@@
	unknown_6:@#
	unknown_7:@
	unknown_8:
identityѕбStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЯЯ*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *M
fHRF
D__inference_decoder_layer_call_and_return_conditional_losses_57670522
StatefulPartitionedCallЁ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ЯЯ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:           
: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:           

!
_user_specified_name	input_2
ќ
Ќ
)__inference_decoder_layer_call_fn_5767234
input_2!
unknown:		@

	unknown_0:@#
	unknown_1:@@
	unknown_2:@#
	unknown_3:@@
	unknown_4:@#
	unknown_5:@@
	unknown_6:@#
	unknown_7:@
	unknown_8:
identityѕбStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЯЯ*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *M
fHRF
D__inference_decoder_layer_call_and_return_conditional_losses_57671862
StatefulPartitionedCallЁ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ЯЯ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:           
: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:           

!
_user_specified_name	input_2
ђ"
═
 __inference__traced_save_5768069
file_prefix3
/savev2_decoder_input_kernel_read_readvariableop1
-savev2_decoder_input_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop.
*savev2_conv2d_4_kernel_read_readvariableop,
(savev2_conv2d_4_bias_read_readvariableop
savev2_const

identity_1ѕбMergeV2CheckpointsЈ
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1І
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardд
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename▒
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*├
value╣BХB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesъ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B 2
SaveV2/shape_and_slicesЧ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0/savev2_decoder_input_kernel_read_readvariableop-savev2_decoder_input_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop*savev2_conv2d_4_kernel_read_readvariableop(savev2_conv2d_4_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesА
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*Ј
_input_shapes~
|: :		@
:@:@@:@:@@:@:@@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:		@
: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,	(
&
_output_shapes
:@: 


_output_shapes
::

_output_shapes
: 
э
■
E__inference_conv2d_3_layer_call_and_return_conditional_losses_5767988

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpЦ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ@*
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpі
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ@2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:         ЯЯ@2
Reluw
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:         ЯЯ@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ЯЯ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:         ЯЯ@
 
_user_specified_nameinputs
Ч%
Ќ
J__inference_decoder_input_layer_call_and_return_conditional_losses_5766892

inputsB
(conv2d_transpose_readvariableop_resource:		@
-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбconv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2В
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2В
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/3ѓ
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2В
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3│
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:		@
*
dtype02!
conv2d_transpose/ReadVariableOp­
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                           @*
paddingSAME*
strides
2
conv2d_transposeї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpц
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           @2
ReluЄ
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+                           @2

IdentityЅ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+                           

 
_user_specified_nameinputs
Ћ
Ќ
*__inference_decoder1_layer_call_fn_5767672

inputs!
unknown:		@

	unknown_0:@#
	unknown_1:@@
	unknown_2:@#
	unknown_3:@@
	unknown_4:@#
	unknown_5:@@
	unknown_6:@#
	unknown_7:@
	unknown_8:
identityѕбStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЯЯ*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_decoder1_layer_call_and_return_conditional_losses_57673332
StatefulPartitionedCallЁ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ЯЯ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:           
: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:           

 
_user_specified_nameinputs
э
■
E__inference_conv2d_1_layer_call_and_return_conditional_losses_5767948

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpЦ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ@*
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpі
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ@2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:         ЯЯ@2
Reluw
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:         ЯЯ@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ЯЯ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:         ЯЯ@
 
_user_specified_nameinputs
с
Э
E__inference_decoder1_layer_call_and_return_conditional_losses_5767333

inputs)
decoder_5767311:		@

decoder_5767313:@)
decoder_5767315:@@
decoder_5767317:@)
decoder_5767319:@@
decoder_5767321:@)
decoder_5767323:@@
decoder_5767325:@)
decoder_5767327:@
decoder_5767329:
identityѕбdecoder/StatefulPartitionedCall╣
decoder/StatefulPartitionedCallStatefulPartitionedCallinputsdecoder_5767311decoder_5767313decoder_5767315decoder_5767317decoder_5767319decoder_5767321decoder_5767323decoder_5767325decoder_5767327decoder_5767329*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЯЯ*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *M
fHRF
D__inference_decoder_layer_call_and_return_conditional_losses_57670522!
decoder/StatefulPartitionedCallЇ
IdentityIdentity(decoder/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ЯЯ2

Identityp
NoOpNoOp ^decoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:           
: : : : : : : : : : 2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall:W S
/
_output_shapes
:           

 
_user_specified_nameinputs
Э
 
E__inference_decoder1_layer_call_and_return_conditional_losses_5767506
decoder_input)
decoder_5767484:		@

decoder_5767486:@)
decoder_5767488:@@
decoder_5767490:@)
decoder_5767492:@@
decoder_5767494:@)
decoder_5767496:@@
decoder_5767498:@)
decoder_5767500:@
decoder_5767502:
identityѕбdecoder/StatefulPartitionedCall└
decoder/StatefulPartitionedCallStatefulPartitionedCalldecoder_inputdecoder_5767484decoder_5767486decoder_5767488decoder_5767490decoder_5767492decoder_5767494decoder_5767496decoder_5767498decoder_5767500decoder_5767502*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЯЯ*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *M
fHRF
D__inference_decoder_layer_call_and_return_conditional_losses_57671862!
decoder/StatefulPartitionedCallЇ
IdentityIdentity(decoder/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ЯЯ2

Identityp
NoOpNoOp ^decoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:           
: : : : : : : : : : 2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall:^ Z
/
_output_shapes
:           

'
_user_specified_namedecoder_input
Э
 
E__inference_decoder1_layer_call_and_return_conditional_losses_5767481
decoder_input)
decoder_5767459:		@

decoder_5767461:@)
decoder_5767463:@@
decoder_5767465:@)
decoder_5767467:@@
decoder_5767469:@)
decoder_5767471:@@
decoder_5767473:@)
decoder_5767475:@
decoder_5767477:
identityѕбdecoder/StatefulPartitionedCall└
decoder/StatefulPartitionedCallStatefulPartitionedCalldecoder_inputdecoder_5767459decoder_5767461decoder_5767463decoder_5767465decoder_5767467decoder_5767469decoder_5767471decoder_5767473decoder_5767475decoder_5767477*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЯЯ*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *M
fHRF
D__inference_decoder_layer_call_and_return_conditional_losses_57670522!
decoder/StatefulPartitionedCallЇ
IdentityIdentity(decoder/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ЯЯ2

Identityp
NoOpNoOp ^decoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:           
: : : : : : : : : : 2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall:^ Z
/
_output_shapes
:           

'
_user_specified_namedecoder_input
§P
┴
D__inference_decoder_layer_call_and_return_conditional_losses_5767811

inputsP
6decoder_input_conv2d_transpose_readvariableop_resource:		@
;
-decoder_input_biasadd_readvariableop_resource:@A
'conv2d_1_conv2d_readvariableop_resource:@@6
(conv2d_1_biasadd_readvariableop_resource:@A
'conv2d_2_conv2d_readvariableop_resource:@@6
(conv2d_2_biasadd_readvariableop_resource:@A
'conv2d_3_conv2d_readvariableop_resource:@@6
(conv2d_3_biasadd_readvariableop_resource:@A
'conv2d_4_conv2d_readvariableop_resource:@6
(conv2d_4_biasadd_readvariableop_resource:
identityѕбconv2d_1/BiasAdd/ReadVariableOpбconv2d_1/Conv2D/ReadVariableOpбconv2d_2/BiasAdd/ReadVariableOpбconv2d_2/Conv2D/ReadVariableOpбconv2d_3/BiasAdd/ReadVariableOpбconv2d_3/Conv2D/ReadVariableOpбconv2d_4/BiasAdd/ReadVariableOpбconv2d_4/Conv2D/ReadVariableOpб$decoder_input/BiasAdd/ReadVariableOpб-decoder_input/conv2d_transpose/ReadVariableOp`
decoder_input/ShapeShapeinputs*
T0*
_output_shapes
:2
decoder_input/Shapeљ
!decoder_input/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!decoder_input/strided_slice/stackћ
#decoder_input/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#decoder_input/strided_slice/stack_1ћ
#decoder_input/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#decoder_input/strided_slice/stack_2Х
decoder_input/strided_sliceStridedSlicedecoder_input/Shape:output:0*decoder_input/strided_slice/stack:output:0,decoder_input/strided_slice/stack_1:output:0,decoder_input/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
decoder_input/strided_sliceq
decoder_input/stack/1Const*
_output_shapes
: *
dtype0*
value
B :Я2
decoder_input/stack/1q
decoder_input/stack/2Const*
_output_shapes
: *
dtype0*
value
B :Я2
decoder_input/stack/2p
decoder_input/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
decoder_input/stack/3Т
decoder_input/stackPack$decoder_input/strided_slice:output:0decoder_input/stack/1:output:0decoder_input/stack/2:output:0decoder_input/stack/3:output:0*
N*
T0*
_output_shapes
:2
decoder_input/stackћ
#decoder_input/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#decoder_input/strided_slice_1/stackў
%decoder_input/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%decoder_input/strided_slice_1/stack_1ў
%decoder_input/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%decoder_input/strided_slice_1/stack_2└
decoder_input/strided_slice_1StridedSlicedecoder_input/stack:output:0,decoder_input/strided_slice_1/stack:output:0.decoder_input/strided_slice_1/stack_1:output:0.decoder_input/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
decoder_input/strided_slice_1П
-decoder_input/conv2d_transpose/ReadVariableOpReadVariableOp6decoder_input_conv2d_transpose_readvariableop_resource*&
_output_shapes
:		@
*
dtype02/
-decoder_input/conv2d_transpose/ReadVariableOpў
decoder_input/conv2d_transposeConv2DBackpropInputdecoder_input/stack:output:05decoder_input/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*1
_output_shapes
:         ЯЯ@*
paddingSAME*
strides
2 
decoder_input/conv2d_transposeХ
$decoder_input/BiasAdd/ReadVariableOpReadVariableOp-decoder_input_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02&
$decoder_input/BiasAdd/ReadVariableOp╠
decoder_input/BiasAddBiasAdd'decoder_input/conv2d_transpose:output:0,decoder_input/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ@2
decoder_input/BiasAddї
decoder_input/ReluReludecoder_input/BiasAdd:output:0*
T0*1
_output_shapes
:         ЯЯ@2
decoder_input/Relu░
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_1/Conv2D/ReadVariableOp┌
conv2d_1/Conv2DConv2D decoder_input/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ@*
paddingSAME*
strides
2
conv2d_1/Conv2DД
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp«
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ@2
conv2d_1/BiasAdd}
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*1
_output_shapes
:         ЯЯ@2
conv2d_1/Relu╝
tf.__operators__.add/AddV2AddV2conv2d_1/Relu:activations:0 decoder_input/Relu:activations:0*
T0*1
_output_shapes
:         ЯЯ@2
tf.__operators__.add/AddV2░
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_2/Conv2D/ReadVariableOpп
conv2d_2/Conv2DConv2Dtf.__operators__.add/AddV2:z:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ@*
paddingSAME*
strides
2
conv2d_2/Conv2DД
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp«
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ@2
conv2d_2/BiasAdd}
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*1
_output_shapes
:         ЯЯ@2
conv2d_2/Relu░
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_3/Conv2D/ReadVariableOpН
conv2d_3/Conv2DConv2Dconv2d_2/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ@*
paddingSAME*
strides
2
conv2d_3/Conv2DД
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp«
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ@2
conv2d_3/BiasAdd}
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*1
_output_shapes
:         ЯЯ@2
conv2d_3/Relu╗
tf.__operators__.add_1/AddV2AddV2conv2d_3/Relu:activations:0conv2d_2/Relu:activations:0*
T0*1
_output_shapes
:         ЯЯ@2
tf.__operators__.add_1/AddV2░
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02 
conv2d_4/Conv2D/ReadVariableOp┌
conv2d_4/Conv2DConv2D tf.__operators__.add_1/AddV2:z:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ*
paddingSAME*
strides
2
conv2d_4/Conv2DД
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp«
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ2
conv2d_4/BiasAddЎ
(tf.clip_by_value/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2*
(tf.clip_by_value/clip_by_value/Minimum/yт
&tf.clip_by_value/clip_by_value/MinimumMinimumconv2d_4/BiasAdd:output:01tf.clip_by_value/clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:         ЯЯ2(
&tf.clip_by_value/clip_by_value/MinimumЅ
 tf.clip_by_value/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 tf.clip_by_value/clip_by_value/yя
tf.clip_by_value/clip_by_valueMaximum*tf.clip_by_value/clip_by_value/Minimum:z:0)tf.clip_by_value/clip_by_value/y:output:0*
T0*1
_output_shapes
:         ЯЯ2 
tf.clip_by_value/clip_by_valueЄ
IdentityIdentity"tf.clip_by_value/clip_by_value:z:0^NoOp*
T0*1
_output_shapes
:         ЯЯ2

Identity▒
NoOpNoOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp%^decoder_input/BiasAdd/ReadVariableOp.^decoder_input/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:           
: : : : : : : : : : 2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2L
$decoder_input/BiasAdd/ReadVariableOp$decoder_input/BiasAdd/ReadVariableOp2^
-decoder_input/conv2d_transpose/ReadVariableOp-decoder_input/conv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:           

 
_user_specified_nameinputs
ф
ъ
*__inference_decoder1_layer_call_fn_5767356
decoder_input!
unknown:		@

	unknown_0:@#
	unknown_1:@@
	unknown_2:@#
	unknown_3:@@
	unknown_4:@#
	unknown_5:@@
	unknown_6:@#
	unknown_7:@
	unknown_8:
identityѕбStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCalldecoder_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЯЯ*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_decoder1_layer_call_and_return_conditional_losses_57673332
StatefulPartitionedCallЁ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ЯЯ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:           
: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
/
_output_shapes
:           

'
_user_specified_namedecoder_input
ќ(
Ю
D__inference_decoder_layer_call_and_return_conditional_losses_5767186

inputs/
decoder_input_5767154:		@
#
decoder_input_5767156:@*
conv2d_1_5767159:@@
conv2d_1_5767161:@*
conv2d_2_5767165:@@
conv2d_2_5767167:@*
conv2d_3_5767170:@@
conv2d_3_5767172:@*
conv2d_4_5767176:@
conv2d_4_5767178:
identityѕб conv2d_1/StatefulPartitionedCallб conv2d_2/StatefulPartitionedCallб conv2d_3/StatefulPartitionedCallб conv2d_4/StatefulPartitionedCallб%decoder_input/StatefulPartitionedCall┐
%decoder_input/StatefulPartitionedCallStatefulPartitionedCallinputsdecoder_input_5767154decoder_input_5767156*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЯЯ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__inference_decoder_input_layer_call_and_return_conditional_losses_57669722'
%decoder_input/StatefulPartitionedCall╬
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall.decoder_input/StatefulPartitionedCall:output:0conv2d_1_5767159conv2d_1_5767161*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЯЯ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_57669892"
 conv2d_1/StatefulPartitionedCallп
tf.__operators__.add/AddV2AddV2)conv2d_1/StatefulPartitionedCall:output:0.decoder_input/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:         ЯЯ@2
tf.__operators__.add/AddV2Й
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCalltf.__operators__.add/AddV2:z:0conv2d_2_5767165conv2d_2_5767167*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЯЯ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_57670072"
 conv2d_2/StatefulPartitionedCall╔
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_5767170conv2d_3_5767172*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЯЯ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_57670242"
 conv2d_3/StatefulPartitionedCallО
tf.__operators__.add_1/AddV2AddV2)conv2d_3/StatefulPartitionedCall:output:0)conv2d_2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:         ЯЯ@2
tf.__operators__.add_1/AddV2└
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_1/AddV2:z:0conv2d_4_5767176conv2d_4_5767178*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЯЯ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_57670412"
 conv2d_4/StatefulPartitionedCallЎ
(tf.clip_by_value/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2*
(tf.clip_by_value/clip_by_value/Minimum/yш
&tf.clip_by_value/clip_by_value/MinimumMinimum)conv2d_4/StatefulPartitionedCall:output:01tf.clip_by_value/clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:         ЯЯ2(
&tf.clip_by_value/clip_by_value/MinimumЅ
 tf.clip_by_value/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 tf.clip_by_value/clip_by_value/yя
tf.clip_by_value/clip_by_valueMaximum*tf.clip_by_value/clip_by_value/Minimum:z:0)tf.clip_by_value/clip_by_value/y:output:0*
T0*1
_output_shapes
:         ЯЯ2 
tf.clip_by_value/clip_by_valueЄ
IdentityIdentity"tf.clip_by_value/clip_by_value:z:0^NoOp*
T0*1
_output_shapes
:         ЯЯ2

Identityѓ
NoOpNoOp!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall&^decoder_input/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:           
: : : : : : : : : : 2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2N
%decoder_input/StatefulPartitionedCall%decoder_input/StatefulPartitionedCall:W S
/
_output_shapes
:           

 
_user_specified_nameinputs
Ч%
Ќ
J__inference_decoder_input_layer_call_and_return_conditional_losses_5767895

inputsB
(conv2d_transpose_readvariableop_resource:		@
-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбconv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2В
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2В
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/3ѓ
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2В
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3│
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:		@
*
dtype02!
conv2d_transpose/ReadVariableOp­
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                           @*
paddingSAME*
strides
2
conv2d_transposeї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpц
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           @2
ReluЄ
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+                           @2

IdentityЅ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+                           

 
_user_specified_nameinputs
Њ
ќ
)__inference_decoder_layer_call_fn_5767861

inputs!
unknown:		@

	unknown_0:@#
	unknown_1:@@
	unknown_2:@#
	unknown_3:@@
	unknown_4:@#
	unknown_5:@@
	unknown_6:@#
	unknown_7:@
	unknown_8:
identityѕбStatefulPartitionedCallв
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЯЯ*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *M
fHRF
D__inference_decoder_layer_call_and_return_conditional_losses_57671862
StatefulPartitionedCallЁ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ЯЯ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:           
: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:           

 
_user_specified_nameinputs
э
■
E__inference_conv2d_2_layer_call_and_return_conditional_losses_5767007

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpЦ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ@*
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpі
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ@2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:         ЯЯ@2
Reluw
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:         ЯЯ@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ЯЯ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:         ЯЯ@
 
_user_specified_nameinputs
Љ
■
E__inference_conv2d_4_layer_call_and_return_conditional_losses_5768007

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpЦ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ*
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpі
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ2	
BiasAddu
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:         ЯЯ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ЯЯ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:         ЯЯ@
 
_user_specified_nameinputs
А
Ќ
J__inference_decoder_input_layer_call_and_return_conditional_losses_5767919

inputsB
(conv2d_transpose_readvariableop_resource:		@
-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбconv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceU
stack/1Const*
_output_shapes
: *
dtype0*
value
B :Я2	
stack/1U
stack/2Const*
_output_shapes
: *
dtype0*
value
B :Я2	
stack/2T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/3њ
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2В
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1│
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:		@
*
dtype02!
conv2d_transpose/ReadVariableOpЯ
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*1
_output_shapes
:         ЯЯ@*
paddingSAME*
strides
2
conv2d_transposeї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpћ
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ@2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:         ЯЯ@2
Reluw
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:         ЯЯ@2

IdentityЅ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:           
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:           

 
_user_specified_nameinputs
Е
Ъ
*__inference_conv2d_2_layer_call_fn_5767977

inputs!
unknown:@@
	unknown_0:@
identityѕбStatefulPartitionedCallё
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЯЯ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_57670072
StatefulPartitionedCallЁ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ЯЯ@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ЯЯ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ЯЯ@
 
_user_specified_nameinputs
Ћ
Ќ
*__inference_decoder1_layer_call_fn_5767697

inputs!
unknown:		@

	unknown_0:@#
	unknown_1:@@
	unknown_2:@#
	unknown_3:@@
	unknown_4:@#
	unknown_5:@@
	unknown_6:@#
	unknown_7:@
	unknown_8:
identityѕбStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЯЯ*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_decoder1_layer_call_and_return_conditional_losses_57674082
StatefulPartitionedCallЁ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ЯЯ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:           
: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:           

 
_user_specified_nameinputs
Ц.
╚
#__inference__traced_restore_5768109
file_prefix?
%assignvariableop_decoder_input_kernel:		@
3
%assignvariableop_1_decoder_input_bias:@<
"assignvariableop_2_conv2d_1_kernel:@@.
 assignvariableop_3_conv2d_1_bias:@<
"assignvariableop_4_conv2d_2_kernel:@@.
 assignvariableop_5_conv2d_2_bias:@<
"assignvariableop_6_conv2d_3_kernel:@@.
 assignvariableop_7_conv2d_3_bias:@<
"assignvariableop_8_conv2d_4_kernel:@.
 assignvariableop_9_conv2d_4_bias:
identity_11ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_2бAssignVariableOp_3бAssignVariableOp_4бAssignVariableOp_5бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9и
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*├
value╣BХB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesц
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B 2
RestoreV2/shape_and_slicesР
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*@
_output_shapes.
,:::::::::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identityц
AssignVariableOpAssignVariableOp%assignvariableop_decoder_input_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1ф
AssignVariableOp_1AssignVariableOp%assignvariableop_1_decoder_input_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Д
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Ц
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Д
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Ц
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6Д
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Ц
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Д
AssignVariableOp_8AssignVariableOp"assignvariableop_8_conv2d_4_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Ц
AssignVariableOp_9AssignVariableOp assignvariableop_9_conv2d_4_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp║
Identity_10Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_10f
Identity_11IdentityIdentity_10:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_11б
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_11Identity_11:output:0*)
_input_shapes
: : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
э
■
E__inference_conv2d_1_layer_call_and_return_conditional_losses_5766989

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpЦ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ@*
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpі
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЯЯ@2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:         ЯЯ@2
Reluw
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:         ЯЯ@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ЯЯ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:         ЯЯ@
 
_user_specified_nameinputs
ф
ъ
*__inference_decoder1_layer_call_fn_5767456
decoder_input!
unknown:		@

	unknown_0:@#
	unknown_1:@@
	unknown_2:@#
	unknown_3:@@
	unknown_4:@#
	unknown_5:@@
	unknown_6:@#
	unknown_7:@
	unknown_8:
identityѕбStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCalldecoder_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЯЯ*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_decoder1_layer_call_and_return_conditional_losses_57674082
StatefulPartitionedCallЁ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ЯЯ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:           
: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
/
_output_shapes
:           

'
_user_specified_namedecoder_input"еL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*╚
serving_default┤
O
decoder_input>
serving_default_decoder_input:0           
E
decoder:
StatefulPartitionedCall:0         ЯЯtensorflow/serving/predict:иЃ
■
layer_with_weights-0
layer-0
	variables
trainable_variables
regularization_losses
	keras_api

signatures
*X&call_and_return_all_conditional_losses
Y_default_save_signature
Z__call__"
_tf_keras_sequential
ъ
layer-0
layer_with_weights-0
layer-1
	layer_with_weights-1
	layer-2

layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
layer-8
	variables
trainable_variables
regularization_losses
	keras_api
*[&call_and_return_all_conditional_losses
\__call__"
_tf_keras_network
f
0
1
2
3
4
5
6
7
8
9"
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
8
9"
trackable_list_wrapper
 "
trackable_list_wrapper
╩
	variables
trainable_variables
regularization_losses
layer_metrics

layers
 metrics
!layer_regularization_losses
"non_trainable_variables
Z__call__
Y_default_save_signature
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
,
]serving_default"
signature_map
"
_tf_keras_input_layer
╗

kernel
bias
#	variables
$trainable_variables
%regularization_losses
&	keras_api
*^&call_and_return_all_conditional_losses
___call__"
_tf_keras_layer
╗

kernel
bias
'	variables
(trainable_variables
)regularization_losses
*	keras_api
*`&call_and_return_all_conditional_losses
a__call__"
_tf_keras_layer
(
+	keras_api"
_tf_keras_layer
╗

kernel
bias
,	variables
-trainable_variables
.regularization_losses
/	keras_api
*b&call_and_return_all_conditional_losses
c__call__"
_tf_keras_layer
╗

kernel
bias
0	variables
1trainable_variables
2regularization_losses
3	keras_api
*d&call_and_return_all_conditional_losses
e__call__"
_tf_keras_layer
(
4	keras_api"
_tf_keras_layer
╗

kernel
bias
5	variables
6trainable_variables
7regularization_losses
8	keras_api
*f&call_and_return_all_conditional_losses
g__call__"
_tf_keras_layer
(
9	keras_api"
_tf_keras_layer
f
0
1
2
3
4
5
6
7
8
9"
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
8
9"
trackable_list_wrapper
 "
trackable_list_wrapper
Г
	variables
trainable_variables
regularization_losses
:layer_metrics

;layers
<metrics
=layer_regularization_losses
>non_trainable_variables
\__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
.:,		@
2decoder_input/kernel
 :@2decoder_input/bias
):'@@2conv2d_1/kernel
:@2conv2d_1/bias
):'@@2conv2d_2/kernel
:@2conv2d_2/bias
):'@@2conv2d_3/kernel
:@2conv2d_3/bias
):'@2conv2d_4/kernel
:2conv2d_4/bias
 "
trackable_dict_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Г
#	variables
$trainable_variables
%regularization_losses
?layer_metrics

@layers
Ametrics
Blayer_regularization_losses
Cnon_trainable_variables
___call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Г
'	variables
(trainable_variables
)regularization_losses
Dlayer_metrics

Elayers
Fmetrics
Glayer_regularization_losses
Hnon_trainable_variables
a__call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Г
,	variables
-trainable_variables
.regularization_losses
Ilayer_metrics

Jlayers
Kmetrics
Llayer_regularization_losses
Mnon_trainable_variables
c__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Г
0	variables
1trainable_variables
2regularization_losses
Nlayer_metrics

Olayers
Pmetrics
Qlayer_regularization_losses
Rnon_trainable_variables
e__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Г
5	variables
6trainable_variables
7regularization_losses
Slayer_metrics

Tlayers
Umetrics
Vlayer_regularization_losses
Wnon_trainable_variables
g__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
 "
trackable_dict_wrapper
_
0
1
	2

3
4
5
6
7
8"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Р2▀
E__inference_decoder1_layer_call_and_return_conditional_losses_5767590
E__inference_decoder1_layer_call_and_return_conditional_losses_5767647
E__inference_decoder1_layer_call_and_return_conditional_losses_5767481
E__inference_decoder1_layer_call_and_return_conditional_losses_5767506└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
МBл
"__inference__wrapped_model_5766854decoder_input"ў
Љ▓Ї
FullArgSpec
argsџ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ш2з
*__inference_decoder1_layer_call_fn_5767356
*__inference_decoder1_layer_call_fn_5767672
*__inference_decoder1_layer_call_fn_5767697
*__inference_decoder1_layer_call_fn_5767456└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
я2█
D__inference_decoder_layer_call_and_return_conditional_losses_5767754
D__inference_decoder_layer_call_and_return_conditional_losses_5767811
D__inference_decoder_layer_call_and_return_conditional_losses_5767269
D__inference_decoder_layer_call_and_return_conditional_losses_5767304└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Ы2№
)__inference_decoder_layer_call_fn_5767075
)__inference_decoder_layer_call_fn_5767836
)__inference_decoder_layer_call_fn_5767861
)__inference_decoder_layer_call_fn_5767234└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
мB¤
%__inference_signature_wrapper_5767533decoder_input"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
└2й
J__inference_decoder_input_layer_call_and_return_conditional_losses_5767895
J__inference_decoder_input_layer_call_and_return_conditional_losses_5767919б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
і2Є
/__inference_decoder_input_layer_call_fn_5767928
/__inference_decoder_input_layer_call_fn_5767937б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
№2В
E__inference_conv2d_1_layer_call_and_return_conditional_losses_5767948б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
н2Л
*__inference_conv2d_1_layer_call_fn_5767957б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
№2В
E__inference_conv2d_2_layer_call_and_return_conditional_losses_5767968б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
н2Л
*__inference_conv2d_2_layer_call_fn_5767977б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
№2В
E__inference_conv2d_3_layer_call_and_return_conditional_losses_5767988б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
н2Л
*__inference_conv2d_3_layer_call_fn_5767997б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
№2В
E__inference_conv2d_4_layer_call_and_return_conditional_losses_5768007б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
н2Л
*__inference_conv2d_4_layer_call_fn_5768016б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 ░
"__inference__wrapped_model_5766854Ѕ
>б;
4б1
/і,
decoder_input           

ф ";ф8
6
decoder+і(
decoder         ЯЯ╣
E__inference_conv2d_1_layer_call_and_return_conditional_losses_5767948p9б6
/б,
*і'
inputs         ЯЯ@
ф "/б,
%і"
0         ЯЯ@
џ Љ
*__inference_conv2d_1_layer_call_fn_5767957c9б6
/б,
*і'
inputs         ЯЯ@
ф ""і         ЯЯ@╣
E__inference_conv2d_2_layer_call_and_return_conditional_losses_5767968p9б6
/б,
*і'
inputs         ЯЯ@
ф "/б,
%і"
0         ЯЯ@
џ Љ
*__inference_conv2d_2_layer_call_fn_5767977c9б6
/б,
*і'
inputs         ЯЯ@
ф ""і         ЯЯ@╣
E__inference_conv2d_3_layer_call_and_return_conditional_losses_5767988p9б6
/б,
*і'
inputs         ЯЯ@
ф "/б,
%і"
0         ЯЯ@
џ Љ
*__inference_conv2d_3_layer_call_fn_5767997c9б6
/б,
*і'
inputs         ЯЯ@
ф ""і         ЯЯ@╣
E__inference_conv2d_4_layer_call_and_return_conditional_losses_5768007p9б6
/б,
*і'
inputs         ЯЯ@
ф "/б,
%і"
0         ЯЯ
џ Љ
*__inference_conv2d_4_layer_call_fn_5768016c9б6
/б,
*і'
inputs         ЯЯ@
ф ""і         ЯЯ¤
E__inference_decoder1_layer_call_and_return_conditional_losses_5767481Ё
FбC
<б9
/і,
decoder_input           

p 

 
ф "/б,
%і"
0         ЯЯ
џ ¤
E__inference_decoder1_layer_call_and_return_conditional_losses_5767506Ё
FбC
<б9
/і,
decoder_input           

p

 
ф "/б,
%і"
0         ЯЯ
џ К
E__inference_decoder1_layer_call_and_return_conditional_losses_5767590~
?б<
5б2
(і%
inputs           

p 

 
ф "/б,
%і"
0         ЯЯ
џ К
E__inference_decoder1_layer_call_and_return_conditional_losses_5767647~
?б<
5б2
(і%
inputs           

p

 
ф "/б,
%і"
0         ЯЯ
џ д
*__inference_decoder1_layer_call_fn_5767356x
FбC
<б9
/і,
decoder_input           

p 

 
ф ""і         ЯЯд
*__inference_decoder1_layer_call_fn_5767456x
FбC
<б9
/і,
decoder_input           

p

 
ф ""і         ЯЯЪ
*__inference_decoder1_layer_call_fn_5767672q
?б<
5б2
(і%
inputs           

p 

 
ф ""і         ЯЯЪ
*__inference_decoder1_layer_call_fn_5767697q
?б<
5б2
(і%
inputs           

p

 
ф ""і         ЯЯ▀
J__inference_decoder_input_layer_call_and_return_conditional_losses_5767895љIбF
?б<
:і7
inputs+                           

ф "?б<
5і2
0+                           @
џ ╝
J__inference_decoder_input_layer_call_and_return_conditional_losses_5767919n7б4
-б*
(і%
inputs           

ф "/б,
%і"
0         ЯЯ@
џ и
/__inference_decoder_input_layer_call_fn_5767928ЃIбF
?б<
:і7
inputs+                           

ф "2і/+                           @ћ
/__inference_decoder_input_layer_call_fn_5767937a7б4
-б*
(і%
inputs           

ф ""і         ЯЯ@К
D__inference_decoder_layer_call_and_return_conditional_losses_5767269
@б=
6б3
)і&
input_2           

p 

 
ф "/б,
%і"
0         ЯЯ
џ К
D__inference_decoder_layer_call_and_return_conditional_losses_5767304
@б=
6б3
)і&
input_2           

p

 
ф "/б,
%і"
0         ЯЯ
џ к
D__inference_decoder_layer_call_and_return_conditional_losses_5767754~
?б<
5б2
(і%
inputs           

p 

 
ф "/б,
%і"
0         ЯЯ
џ к
D__inference_decoder_layer_call_and_return_conditional_losses_5767811~
?б<
5б2
(і%
inputs           

p

 
ф "/б,
%і"
0         ЯЯ
џ Ъ
)__inference_decoder_layer_call_fn_5767075r
@б=
6б3
)і&
input_2           

p 

 
ф ""і         ЯЯЪ
)__inference_decoder_layer_call_fn_5767234r
@б=
6б3
)і&
input_2           

p

 
ф ""і         ЯЯъ
)__inference_decoder_layer_call_fn_5767836q
?б<
5б2
(і%
inputs           

p 

 
ф ""і         ЯЯъ
)__inference_decoder_layer_call_fn_5767861q
?б<
5б2
(і%
inputs           

p

 
ф ""і         ЯЯ─
%__inference_signature_wrapper_5767533џ
OбL
б 
EфB
@
decoder_input/і,
decoder_input           
";ф8
6
decoder+і(
decoder         ЯЯ