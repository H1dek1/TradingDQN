??
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
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
dtypetype?
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
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
?
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
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
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
<
Sub
x"T
y"T
z"T"
Ttype:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.6.02v2.6.0-rc2-32-g919f693420e8ݚ
x
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_3/kernel
q
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes

:*
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:*
dtype0
x
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_4/kernel
q
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes

:*
dtype0
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
:*
dtype0
x
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_5/kernel
q
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes

:*
dtype0
p
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
:*
dtype0
?
state_value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*#
shared_namestate_value/kernel
y
&state_value/kernel/Read/ReadVariableOpReadVariableOpstate_value/kernel*
_output_shapes

:*
dtype0
x
state_value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namestate_value/bias
q
$state_value/bias/Read/ReadVariableOpReadVariableOpstate_value/bias*
_output_shapes
:*
dtype0
|
advantage/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_nameadvantage/kernel
u
$advantage/kernel/Read/ReadVariableOpReadVariableOpadvantage/kernel*
_output_shapes

:*
dtype0
t
advantage/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameadvantage/bias
m
"advantage/bias/Read/ReadVariableOpReadVariableOpadvantage/bias*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0

NoOpNoOp
?#
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?"
value?"B?" B?"
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer-8

layer-9
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
 
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
 
R
	variables
regularization_losses
trainable_variables
 	keras_api
h

!kernel
"bias
#	variables
$regularization_losses
%trainable_variables
&	keras_api
h

'kernel
(bias
)	variables
*regularization_losses
+trainable_variables
,	keras_api
h

-kernel
.bias
/	variables
0regularization_losses
1trainable_variables
2	keras_api
R
3	variables
4regularization_losses
5trainable_variables
6	keras_api
R
7	variables
8regularization_losses
9trainable_variables
:	keras_api
 
F
0
1
2
3
!4
"5
'6
(7
-8
.9
 
F
0
1
2
3
!4
"5
'6
(7
-8
.9
?
	variables

;layers
<non_trainable_variables
=metrics
>layer_metrics
?layer_regularization_losses
regularization_losses
trainable_variables
 
ZX
VARIABLE_VALUEdense_3/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_3/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
	variables

@layers
Anon_trainable_variables
Bmetrics
Clayer_metrics
Dlayer_regularization_losses
regularization_losses
trainable_variables
ZX
VARIABLE_VALUEdense_4/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_4/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
	variables

Elayers
Fnon_trainable_variables
Gmetrics
Hlayer_metrics
Ilayer_regularization_losses
regularization_losses
trainable_variables
 
 
 
?
	variables

Jlayers
Knon_trainable_variables
Lmetrics
Mlayer_metrics
Nlayer_regularization_losses
regularization_losses
trainable_variables
ZX
VARIABLE_VALUEdense_5/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_5/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

!0
"1
 

!0
"1
?
#	variables

Olayers
Pnon_trainable_variables
Qmetrics
Rlayer_metrics
Slayer_regularization_losses
$regularization_losses
%trainable_variables
^\
VARIABLE_VALUEstate_value/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEstate_value/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

'0
(1
 

'0
(1
?
)	variables

Tlayers
Unon_trainable_variables
Vmetrics
Wlayer_metrics
Xlayer_regularization_losses
*regularization_losses
+trainable_variables
\Z
VARIABLE_VALUEadvantage/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEadvantage/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

-0
.1
 

-0
.1
?
/	variables

Ylayers
Znon_trainable_variables
[metrics
\layer_metrics
]layer_regularization_losses
0regularization_losses
1trainable_variables
 
 
 
?
3	variables

^layers
_non_trainable_variables
`metrics
alayer_metrics
blayer_regularization_losses
4regularization_losses
5trainable_variables
 
 
 
?
7	variables

clayers
dnon_trainable_variables
emetrics
flayer_metrics
glayer_regularization_losses
8regularization_losses
9trainable_variables
F
0
1
2
3
4
5
6
7
	8

9
 

h0
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
4
	itotal
	jcount
k	variables
l	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

i0
j1

k	variables
~
serving_default_latent_gainPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
~
serving_default_series_dataPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_latent_gainserving_default_series_datadense_3/kerneldense_3/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/biasstate_value/kernelstate_value/biasadvantage/kerneladvantage/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? */
f*R(
&__inference_signature_wrapper_33059516
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOp&state_value/kernel/Read/ReadVariableOp$state_value/bias/Read/ReadVariableOp$advantage/kernel/Read/ReadVariableOp"advantage/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_save_33059926
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_3/kerneldense_3/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/biasstate_value/kernelstate_value/biasadvantage/kerneladvantage/biastotalcount*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference__traced_restore_33059972??
?[
?	
#__inference__wrapped_model_33059056
series_data
latent_gain@
.model_1_dense_3_matmul_readvariableop_resource:=
/model_1_dense_3_biasadd_readvariableop_resource:@
.model_1_dense_4_matmul_readvariableop_resource:=
/model_1_dense_4_biasadd_readvariableop_resource:@
.model_1_dense_5_matmul_readvariableop_resource:=
/model_1_dense_5_biasadd_readvariableop_resource:D
2model_1_state_value_matmul_readvariableop_resource:A
3model_1_state_value_biasadd_readvariableop_resource:B
0model_1_advantage_matmul_readvariableop_resource:?
1model_1_advantage_biasadd_readvariableop_resource:
identity??(model_1/advantage/BiasAdd/ReadVariableOp?'model_1/advantage/MatMul/ReadVariableOp?&model_1/dense_3/BiasAdd/ReadVariableOp?%model_1/dense_3/MatMul/ReadVariableOp?&model_1/dense_4/BiasAdd/ReadVariableOp?%model_1/dense_4/MatMul/ReadVariableOp?&model_1/dense_5/BiasAdd/ReadVariableOp?%model_1/dense_5/MatMul/ReadVariableOp?*model_1/state_value/BiasAdd/ReadVariableOp?)model_1/state_value/MatMul/ReadVariableOp?
%model_1/dense_3/MatMul/ReadVariableOpReadVariableOp.model_1_dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype02'
%model_1/dense_3/MatMul/ReadVariableOp?
model_1/dense_3/MatMulMatMulseries_data-model_1/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_1/dense_3/MatMul?
&model_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&model_1/dense_3/BiasAdd/ReadVariableOp?
model_1/dense_3/BiasAddBiasAdd model_1/dense_3/MatMul:product:0.model_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_1/dense_3/BiasAdd?
model_1/dense_3/ReluRelu model_1/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_1/dense_3/Relu?
%model_1/dense_4/MatMul/ReadVariableOpReadVariableOp.model_1_dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype02'
%model_1/dense_4/MatMul/ReadVariableOp?
model_1/dense_4/MatMulMatMul"model_1/dense_3/Relu:activations:0-model_1/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_1/dense_4/MatMul?
&model_1/dense_4/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&model_1/dense_4/BiasAdd/ReadVariableOp?
model_1/dense_4/BiasAddBiasAdd model_1/dense_4/MatMul:product:0.model_1/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_1/dense_4/BiasAdd?
model_1/dense_4/ReluRelu model_1/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_1/dense_4/Relu?
!model_1/concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!model_1/concatenate_2/concat/axis?
model_1/concatenate_2/concatConcatV2"model_1/dense_4/Relu:activations:0latent_gain*model_1/concatenate_2/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
model_1/concatenate_2/concat?
%model_1/dense_5/MatMul/ReadVariableOpReadVariableOp.model_1_dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype02'
%model_1/dense_5/MatMul/ReadVariableOp?
model_1/dense_5/MatMulMatMul%model_1/concatenate_2/concat:output:0-model_1/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_1/dense_5/MatMul?
&model_1/dense_5/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&model_1/dense_5/BiasAdd/ReadVariableOp?
model_1/dense_5/BiasAddBiasAdd model_1/dense_5/MatMul:product:0.model_1/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_1/dense_5/BiasAdd?
model_1/dense_5/ReluRelu model_1/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_1/dense_5/Relu?
)model_1/state_value/MatMul/ReadVariableOpReadVariableOp2model_1_state_value_matmul_readvariableop_resource*
_output_shapes

:*
dtype02+
)model_1/state_value/MatMul/ReadVariableOp?
model_1/state_value/MatMulMatMul"model_1/dense_5/Relu:activations:01model_1/state_value/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_1/state_value/MatMul?
*model_1/state_value/BiasAdd/ReadVariableOpReadVariableOp3model_1_state_value_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*model_1/state_value/BiasAdd/ReadVariableOp?
model_1/state_value/BiasAddBiasAdd$model_1/state_value/MatMul:product:02model_1/state_value/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_1/state_value/BiasAdd?
'model_1/advantage/MatMul/ReadVariableOpReadVariableOp0model_1_advantage_matmul_readvariableop_resource*
_output_shapes

:*
dtype02)
'model_1/advantage/MatMul/ReadVariableOp?
model_1/advantage/MatMulMatMul"model_1/dense_5/Relu:activations:0/model_1/advantage/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_1/advantage/MatMul?
(model_1/advantage/BiasAdd/ReadVariableOpReadVariableOp1model_1_advantage_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_1/advantage/BiasAdd/ReadVariableOp?
model_1/advantage/BiasAddBiasAdd"model_1/advantage/MatMul:product:00model_1/advantage/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_1/advantage/BiasAdd?
!model_1/concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!model_1/concatenate_3/concat/axis?
model_1/concatenate_3/concatConcatV2$model_1/state_value/BiasAdd:output:0"model_1/advantage/BiasAdd:output:0*model_1/concatenate_3/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
model_1/concatenate_3/concat?
#model_1/Q_value/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2%
#model_1/Q_value/strided_slice/stack?
%model_1/Q_value/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2'
%model_1/Q_value/strided_slice/stack_1?
%model_1/Q_value/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%model_1/Q_value/strided_slice/stack_2?
model_1/Q_value/strided_sliceStridedSlice%model_1/concatenate_3/concat:output:0,model_1/Q_value/strided_slice/stack:output:0.model_1/Q_value/strided_slice/stack_1:output:0.model_1/Q_value/strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
model_1/Q_value/strided_slice?
model_1/Q_value/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
model_1/Q_value/ExpandDims/dim?
model_1/Q_value/ExpandDims
ExpandDims&model_1/Q_value/strided_slice:output:0'model_1/Q_value/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2
model_1/Q_value/ExpandDims?
%model_1/Q_value/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2'
%model_1/Q_value/strided_slice_1/stack?
'model_1/Q_value/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'model_1/Q_value/strided_slice_1/stack_1?
'model_1/Q_value/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'model_1/Q_value/strided_slice_1/stack_2?
model_1/Q_value/strided_slice_1StridedSlice%model_1/concatenate_3/concat:output:0.model_1/Q_value/strided_slice_1/stack:output:00model_1/Q_value/strided_slice_1/stack_1:output:00model_1/Q_value/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2!
model_1/Q_value/strided_slice_1?
model_1/Q_value/addAddV2#model_1/Q_value/ExpandDims:output:0(model_1/Q_value/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2
model_1/Q_value/add?
%model_1/Q_value/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2'
%model_1/Q_value/strided_slice_2/stack?
'model_1/Q_value/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'model_1/Q_value/strided_slice_2/stack_1?
'model_1/Q_value/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'model_1/Q_value/strided_slice_2/stack_2?
model_1/Q_value/strided_slice_2StridedSlice%model_1/concatenate_3/concat:output:0.model_1/Q_value/strided_slice_2/stack:output:00model_1/Q_value/strided_slice_2/stack_1:output:00model_1/Q_value/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2!
model_1/Q_value/strided_slice_2?
&model_1/Q_value/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2(
&model_1/Q_value/Mean/reduction_indices?
model_1/Q_value/MeanMean(model_1/Q_value/strided_slice_2:output:0/model_1/Q_value/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2
model_1/Q_value/Mean?
model_1/Q_value/subSubmodel_1/Q_value/add:z:0model_1/Q_value/Mean:output:0*
T0*'
_output_shapes
:?????????2
model_1/Q_value/subr
IdentityIdentitymodel_1/Q_value/sub:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp)^model_1/advantage/BiasAdd/ReadVariableOp(^model_1/advantage/MatMul/ReadVariableOp'^model_1/dense_3/BiasAdd/ReadVariableOp&^model_1/dense_3/MatMul/ReadVariableOp'^model_1/dense_4/BiasAdd/ReadVariableOp&^model_1/dense_4/MatMul/ReadVariableOp'^model_1/dense_5/BiasAdd/ReadVariableOp&^model_1/dense_5/MatMul/ReadVariableOp+^model_1/state_value/BiasAdd/ReadVariableOp*^model_1/state_value/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::?????????:?????????: : : : : : : : : : 2T
(model_1/advantage/BiasAdd/ReadVariableOp(model_1/advantage/BiasAdd/ReadVariableOp2R
'model_1/advantage/MatMul/ReadVariableOp'model_1/advantage/MatMul/ReadVariableOp2P
&model_1/dense_3/BiasAdd/ReadVariableOp&model_1/dense_3/BiasAdd/ReadVariableOp2N
%model_1/dense_3/MatMul/ReadVariableOp%model_1/dense_3/MatMul/ReadVariableOp2P
&model_1/dense_4/BiasAdd/ReadVariableOp&model_1/dense_4/BiasAdd/ReadVariableOp2N
%model_1/dense_4/MatMul/ReadVariableOp%model_1/dense_4/MatMul/ReadVariableOp2P
&model_1/dense_5/BiasAdd/ReadVariableOp&model_1/dense_5/BiasAdd/ReadVariableOp2N
%model_1/dense_5/MatMul/ReadVariableOp%model_1/dense_5/MatMul/ReadVariableOp2X
*model_1/state_value/BiasAdd/ReadVariableOp*model_1/state_value/BiasAdd/ReadVariableOp2V
)model_1/state_value/MatMul/ReadVariableOp)model_1/state_value/MatMul/ReadVariableOp:T P
'
_output_shapes
:?????????
%
_user_specified_nameseries_data:TP
'
_output_shapes
:?????????
%
_user_specified_namelatent_gain
?
F
*__inference_Q_value_layer_call_fn_33059822

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_Q_value_layer_call_and_return_conditional_losses_330592442
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?N
?
E__inference_model_1_layer_call_and_return_conditional_losses_33059688
inputs_0
inputs_18
&dense_3_matmul_readvariableop_resource:5
'dense_3_biasadd_readvariableop_resource:8
&dense_4_matmul_readvariableop_resource:5
'dense_4_biasadd_readvariableop_resource:8
&dense_5_matmul_readvariableop_resource:5
'dense_5_biasadd_readvariableop_resource:<
*state_value_matmul_readvariableop_resource:9
+state_value_biasadd_readvariableop_resource::
(advantage_matmul_readvariableop_resource:7
)advantage_biasadd_readvariableop_resource:
identity?? advantage/BiasAdd/ReadVariableOp?advantage/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?"state_value/BiasAdd/ReadVariableOp?!state_value/MatMul/ReadVariableOp?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMulinputs_0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/BiasAddp
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_3/Relu?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMuldense_3/Relu:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_4/BiasAddp
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_4/Relux
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_2/concat/axis?
concatenate_2/concatConcatV2dense_4/Relu:activations:0inputs_1"concatenate_2/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatenate_2/concat?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMulconcatenate_2/concat:output:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/BiasAddp
dense_5/ReluReludense_5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_5/Relu?
!state_value/MatMul/ReadVariableOpReadVariableOp*state_value_matmul_readvariableop_resource*
_output_shapes

:*
dtype02#
!state_value/MatMul/ReadVariableOp?
state_value/MatMulMatMuldense_5/Relu:activations:0)state_value/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
state_value/MatMul?
"state_value/BiasAdd/ReadVariableOpReadVariableOp+state_value_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"state_value/BiasAdd/ReadVariableOp?
state_value/BiasAddBiasAddstate_value/MatMul:product:0*state_value/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
state_value/BiasAdd?
advantage/MatMul/ReadVariableOpReadVariableOp(advantage_matmul_readvariableop_resource*
_output_shapes

:*
dtype02!
advantage/MatMul/ReadVariableOp?
advantage/MatMulMatMuldense_5/Relu:activations:0'advantage/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
advantage/MatMul?
 advantage/BiasAdd/ReadVariableOpReadVariableOp)advantage_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 advantage/BiasAdd/ReadVariableOp?
advantage/BiasAddBiasAddadvantage/MatMul:product:0(advantage/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
advantage/BiasAddx
concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_3/concat/axis?
concatenate_3/concatConcatV2state_value/BiasAdd:output:0advantage/BiasAdd:output:0"concatenate_3/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatenate_3/concat?
Q_value/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
Q_value/strided_slice/stack?
Q_value/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
Q_value/strided_slice/stack_1?
Q_value/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
Q_value/strided_slice/stack_2?
Q_value/strided_sliceStridedSliceconcatenate_3/concat:output:0$Q_value/strided_slice/stack:output:0&Q_value/strided_slice/stack_1:output:0&Q_value/strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
Q_value/strided_slice{
Q_value/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
Q_value/ExpandDims/dim?
Q_value/ExpandDims
ExpandDimsQ_value/strided_slice:output:0Q_value/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2
Q_value/ExpandDims?
Q_value/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
Q_value/strided_slice_1/stack?
Q_value/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2!
Q_value/strided_slice_1/stack_1?
Q_value/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
Q_value/strided_slice_1/stack_2?
Q_value/strided_slice_1StridedSliceconcatenate_3/concat:output:0&Q_value/strided_slice_1/stack:output:0(Q_value/strided_slice_1/stack_1:output:0(Q_value/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
Q_value/strided_slice_1?
Q_value/addAddV2Q_value/ExpandDims:output:0 Q_value/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2
Q_value/add?
Q_value/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
Q_value/strided_slice_2/stack?
Q_value/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2!
Q_value/strided_slice_2/stack_1?
Q_value/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
Q_value/strided_slice_2/stack_2?
Q_value/strided_slice_2StridedSliceconcatenate_3/concat:output:0&Q_value/strided_slice_2/stack:output:0(Q_value/strided_slice_2/stack_1:output:0(Q_value/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
Q_value/strided_slice_2?
Q_value/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2 
Q_value/Mean/reduction_indices?
Q_value/MeanMean Q_value/strided_slice_2:output:0'Q_value/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2
Q_value/Mean{
Q_value/subSubQ_value/add:z:0Q_value/Mean:output:0*
T0*'
_output_shapes
:?????????2
Q_value/subj
IdentityIdentityQ_value/sub:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp!^advantage/BiasAdd/ReadVariableOp ^advantage/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp#^state_value/BiasAdd/ReadVariableOp"^state_value/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::?????????:?????????: : : : : : : : : : 2D
 advantage/BiasAdd/ReadVariableOp advantage/BiasAdd/ReadVariableOp2B
advantage/MatMul/ReadVariableOpadvantage/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2H
"state_value/BiasAdd/ReadVariableOp"state_value/BiasAdd/ReadVariableOp2F
!state_value/MatMul/ReadVariableOp!state_value/MatMul/ReadVariableOp:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
w
K__inference_concatenate_2_layer_call_and_return_conditional_losses_33059741
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?

?
G__inference_advantage_layer_call_and_return_conditional_losses_33059151

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
*__inference_model_1_layer_call_fn_33059422
series_data
latent_gain
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallseries_datalatent_gainunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_model_1_layer_call_and_return_conditional_losses_330593732
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::?????????:?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:?????????
%
_user_specified_nameseries_data:TP
'
_output_shapes
:?????????
%
_user_specified_namelatent_gain
?
?
,__inference_advantage_layer_call_fn_33059789

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_advantage_layer_call_and_return_conditional_losses_330591512
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
E__inference_dense_4_layer_call_and_return_conditional_losses_33059093

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
E__inference_dense_5_layer_call_and_return_conditional_losses_33059119

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
E__inference_dense_4_layer_call_and_return_conditional_losses_33059728

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
w
K__inference_concatenate_3_layer_call_and_return_conditional_losses_33059812
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
?
&__inference_signature_wrapper_33059516
latent_gain
series_data
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallseries_datalatent_gainunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__wrapped_model_330590562
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::?????????:?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:?????????
%
_user_specified_namelatent_gain:TP
'
_output_shapes
:?????????
%
_user_specified_nameseries_data
?
F
*__inference_Q_value_layer_call_fn_33059817

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_Q_value_layer_call_and_return_conditional_losses_330591882
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
*__inference_model_1_layer_call_fn_33059214
series_data
latent_gain
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallseries_datalatent_gainunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_model_1_layer_call_and_return_conditional_losses_330591912
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::?????????:?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:?????????
%
_user_specified_nameseries_data:TP
'
_output_shapes
:?????????
%
_user_specified_namelatent_gain
?
?
E__inference_dense_3_layer_call_and_return_conditional_losses_33059076

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
u
K__inference_concatenate_3_layer_call_and_return_conditional_losses_33059164

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?6
?
$__inference__traced_restore_33059972
file_prefix1
assignvariableop_dense_3_kernel:-
assignvariableop_1_dense_3_bias:3
!assignvariableop_2_dense_4_kernel:-
assignvariableop_3_dense_4_bias:3
!assignvariableop_4_dense_5_kernel:-
assignvariableop_5_dense_5_bias:7
%assignvariableop_6_state_value_kernel:1
#assignvariableop_7_state_value_bias:5
#assignvariableop_8_advantage_kernel:/
!assignvariableop_9_advantage_bias:#
assignvariableop_10_total: #
assignvariableop_11_count: 
identity_13??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*H
_output_shapes6
4:::::::::::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_dense_3_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_3_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_4_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_4_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_5_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_5_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp%assignvariableop_6_state_value_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp#assignvariableop_7_state_value_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp#assignvariableop_8_advantage_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp!assignvariableop_9_advantage_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_totalIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_countIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_119
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_12Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_12f
Identity_13IdentityIdentity_12:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_13?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_13Identity_13:output:0*-
_input_shapes
: : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112(
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
?'
?
E__inference_model_1_layer_call_and_return_conditional_losses_33059191

inputs
inputs_1"
dense_3_33059077:
dense_3_33059079:"
dense_4_33059094:
dense_4_33059096:"
dense_5_33059120:
dense_5_33059122:&
state_value_33059136:"
state_value_33059138:$
advantage_33059152: 
advantage_33059154:
identity??!advantage/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?#state_value/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCallinputsdense_3_33059077dense_3_33059079*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_330590762!
dense_3/StatefulPartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_33059094dense_4_33059096*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_4_layer_call_and_return_conditional_losses_330590932!
dense_4/StatefulPartitionedCall?
concatenate_2/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_concatenate_2_layer_call_and_return_conditional_losses_330591062
concatenate_2/PartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_5_33059120dense_5_33059122*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_5_layer_call_and_return_conditional_losses_330591192!
dense_5/StatefulPartitionedCall?
#state_value/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0state_value_33059136state_value_33059138*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_state_value_layer_call_and_return_conditional_losses_330591352%
#state_value/StatefulPartitionedCall?
!advantage/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0advantage_33059152advantage_33059154*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_advantage_layer_call_and_return_conditional_losses_330591512#
!advantage/StatefulPartitionedCall?
concatenate_3/PartitionedCallPartitionedCall,state_value/StatefulPartitionedCall:output:0*advantage/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_concatenate_3_layer_call_and_return_conditional_losses_330591642
concatenate_3/PartitionedCall?
Q_value/PartitionedCallPartitionedCall&concatenate_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_Q_value_layer_call_and_return_conditional_losses_330591882
Q_value/PartitionedCall{
IdentityIdentity Q_value/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp"^advantage/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall$^state_value/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::?????????:?????????: : : : : : : : : : 2F
!advantage/StatefulPartitionedCall!advantage/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2J
#state_value/StatefulPartitionedCall#state_value/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?(
?
E__inference_model_1_layer_call_and_return_conditional_losses_33059455
series_data
latent_gain"
dense_3_33059426:
dense_3_33059428:"
dense_4_33059431:
dense_4_33059433:"
dense_5_33059437:
dense_5_33059439:&
state_value_33059442:"
state_value_33059444:$
advantage_33059447: 
advantage_33059449:
identity??!advantage/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?#state_value/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCallseries_datadense_3_33059426dense_3_33059428*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_330590762!
dense_3/StatefulPartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_33059431dense_4_33059433*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_4_layer_call_and_return_conditional_losses_330590932!
dense_4/StatefulPartitionedCall?
concatenate_2/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0latent_gain*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_concatenate_2_layer_call_and_return_conditional_losses_330591062
concatenate_2/PartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_5_33059437dense_5_33059439*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_5_layer_call_and_return_conditional_losses_330591192!
dense_5/StatefulPartitionedCall?
#state_value/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0state_value_33059442state_value_33059444*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_state_value_layer_call_and_return_conditional_losses_330591352%
#state_value/StatefulPartitionedCall?
!advantage/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0advantage_33059447advantage_33059449*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_advantage_layer_call_and_return_conditional_losses_330591512#
!advantage/StatefulPartitionedCall?
concatenate_3/PartitionedCallPartitionedCall,state_value/StatefulPartitionedCall:output:0*advantage/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_concatenate_3_layer_call_and_return_conditional_losses_330591642
concatenate_3/PartitionedCall?
Q_value/PartitionedCallPartitionedCall&concatenate_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_Q_value_layer_call_and_return_conditional_losses_330591882
Q_value/PartitionedCall{
IdentityIdentity Q_value/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp"^advantage/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall$^state_value/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::?????????:?????????: : : : : : : : : : 2F
!advantage/StatefulPartitionedCall!advantage/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2J
#state_value/StatefulPartitionedCall#state_value/StatefulPartitionedCall:T P
'
_output_shapes
:?????????
%
_user_specified_nameseries_data:TP
'
_output_shapes
:?????????
%
_user_specified_namelatent_gain
?
a
E__inference_Q_value_layer_call_and_return_conditional_losses_33059188

inputs
identity{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slicek
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims/dim?

ExpandDims
ExpandDimsstrided_slice:output:0ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2

ExpandDims
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice_1t
addAddV2ExpandDims:output:0strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2
add
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceinputsstrided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice_2r
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indices?
MeanMeanstrided_slice_2:output:0Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2
Mean[
subSubadd:z:0Mean:output:0*
T0*'
_output_shapes
:?????????2
sub[
IdentityIdentitysub:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
G__inference_advantage_layer_call_and_return_conditional_losses_33059799

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
*__inference_model_1_layer_call_fn_33059542
inputs_0
inputs_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_model_1_layer_call_and_return_conditional_losses_330591912
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::?????????:?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
u
K__inference_concatenate_2_layer_call_and_return_conditional_losses_33059106

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
*__inference_dense_4_layer_call_fn_33059717

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_4_layer_call_and_return_conditional_losses_330590932
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
\
0__inference_concatenate_2_layer_call_fn_33059734
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_concatenate_2_layer_call_and_return_conditional_losses_330591062
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?'
?
E__inference_model_1_layer_call_and_return_conditional_losses_33059373

inputs
inputs_1"
dense_3_33059344:
dense_3_33059346:"
dense_4_33059349:
dense_4_33059351:"
dense_5_33059355:
dense_5_33059357:&
state_value_33059360:"
state_value_33059362:$
advantage_33059365: 
advantage_33059367:
identity??!advantage/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?#state_value/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCallinputsdense_3_33059344dense_3_33059346*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_330590762!
dense_3/StatefulPartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_33059349dense_4_33059351*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_4_layer_call_and_return_conditional_losses_330590932!
dense_4/StatefulPartitionedCall?
concatenate_2/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_concatenate_2_layer_call_and_return_conditional_losses_330591062
concatenate_2/PartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_5_33059355dense_5_33059357*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_5_layer_call_and_return_conditional_losses_330591192!
dense_5/StatefulPartitionedCall?
#state_value/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0state_value_33059360state_value_33059362*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_state_value_layer_call_and_return_conditional_losses_330591352%
#state_value/StatefulPartitionedCall?
!advantage/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0advantage_33059365advantage_33059367*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_advantage_layer_call_and_return_conditional_losses_330591512#
!advantage/StatefulPartitionedCall?
concatenate_3/PartitionedCallPartitionedCall,state_value/StatefulPartitionedCall:output:0*advantage/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_concatenate_3_layer_call_and_return_conditional_losses_330591642
concatenate_3/PartitionedCall?
Q_value/PartitionedCallPartitionedCall&concatenate_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_Q_value_layer_call_and_return_conditional_losses_330592442
Q_value/PartitionedCall{
IdentityIdentity Q_value/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp"^advantage/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall$^state_value/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::?????????:?????????: : : : : : : : : : 2F
!advantage/StatefulPartitionedCall!advantage/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2J
#state_value/StatefulPartitionedCall#state_value/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
.__inference_state_value_layer_call_fn_33059770

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_state_value_layer_call_and_return_conditional_losses_330591352
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?N
?
E__inference_model_1_layer_call_and_return_conditional_losses_33059628
inputs_0
inputs_18
&dense_3_matmul_readvariableop_resource:5
'dense_3_biasadd_readvariableop_resource:8
&dense_4_matmul_readvariableop_resource:5
'dense_4_biasadd_readvariableop_resource:8
&dense_5_matmul_readvariableop_resource:5
'dense_5_biasadd_readvariableop_resource:<
*state_value_matmul_readvariableop_resource:9
+state_value_biasadd_readvariableop_resource::
(advantage_matmul_readvariableop_resource:7
)advantage_biasadd_readvariableop_resource:
identity?? advantage/BiasAdd/ReadVariableOp?advantage/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?"state_value/BiasAdd/ReadVariableOp?!state_value/MatMul/ReadVariableOp?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMulinputs_0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/BiasAddp
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_3/Relu?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMuldense_3/Relu:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_4/BiasAddp
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_4/Relux
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_2/concat/axis?
concatenate_2/concatConcatV2dense_4/Relu:activations:0inputs_1"concatenate_2/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatenate_2/concat?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMulconcatenate_2/concat:output:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/BiasAddp
dense_5/ReluReludense_5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_5/Relu?
!state_value/MatMul/ReadVariableOpReadVariableOp*state_value_matmul_readvariableop_resource*
_output_shapes

:*
dtype02#
!state_value/MatMul/ReadVariableOp?
state_value/MatMulMatMuldense_5/Relu:activations:0)state_value/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
state_value/MatMul?
"state_value/BiasAdd/ReadVariableOpReadVariableOp+state_value_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"state_value/BiasAdd/ReadVariableOp?
state_value/BiasAddBiasAddstate_value/MatMul:product:0*state_value/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
state_value/BiasAdd?
advantage/MatMul/ReadVariableOpReadVariableOp(advantage_matmul_readvariableop_resource*
_output_shapes

:*
dtype02!
advantage/MatMul/ReadVariableOp?
advantage/MatMulMatMuldense_5/Relu:activations:0'advantage/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
advantage/MatMul?
 advantage/BiasAdd/ReadVariableOpReadVariableOp)advantage_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 advantage/BiasAdd/ReadVariableOp?
advantage/BiasAddBiasAddadvantage/MatMul:product:0(advantage/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
advantage/BiasAddx
concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_3/concat/axis?
concatenate_3/concatConcatV2state_value/BiasAdd:output:0advantage/BiasAdd:output:0"concatenate_3/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatenate_3/concat?
Q_value/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
Q_value/strided_slice/stack?
Q_value/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
Q_value/strided_slice/stack_1?
Q_value/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
Q_value/strided_slice/stack_2?
Q_value/strided_sliceStridedSliceconcatenate_3/concat:output:0$Q_value/strided_slice/stack:output:0&Q_value/strided_slice/stack_1:output:0&Q_value/strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
Q_value/strided_slice{
Q_value/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
Q_value/ExpandDims/dim?
Q_value/ExpandDims
ExpandDimsQ_value/strided_slice:output:0Q_value/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2
Q_value/ExpandDims?
Q_value/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
Q_value/strided_slice_1/stack?
Q_value/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2!
Q_value/strided_slice_1/stack_1?
Q_value/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
Q_value/strided_slice_1/stack_2?
Q_value/strided_slice_1StridedSliceconcatenate_3/concat:output:0&Q_value/strided_slice_1/stack:output:0(Q_value/strided_slice_1/stack_1:output:0(Q_value/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
Q_value/strided_slice_1?
Q_value/addAddV2Q_value/ExpandDims:output:0 Q_value/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2
Q_value/add?
Q_value/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
Q_value/strided_slice_2/stack?
Q_value/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2!
Q_value/strided_slice_2/stack_1?
Q_value/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
Q_value/strided_slice_2/stack_2?
Q_value/strided_slice_2StridedSliceconcatenate_3/concat:output:0&Q_value/strided_slice_2/stack:output:0(Q_value/strided_slice_2/stack_1:output:0(Q_value/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
Q_value/strided_slice_2?
Q_value/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2 
Q_value/Mean/reduction_indices?
Q_value/MeanMean Q_value/strided_slice_2:output:0'Q_value/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2
Q_value/Mean{
Q_value/subSubQ_value/add:z:0Q_value/Mean:output:0*
T0*'
_output_shapes
:?????????2
Q_value/subj
IdentityIdentityQ_value/sub:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp!^advantage/BiasAdd/ReadVariableOp ^advantage/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp#^state_value/BiasAdd/ReadVariableOp"^state_value/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::?????????:?????????: : : : : : : : : : 2D
 advantage/BiasAdd/ReadVariableOp advantage/BiasAdd/ReadVariableOp2B
advantage/MatMul/ReadVariableOpadvantage/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2H
"state_value/BiasAdd/ReadVariableOp"state_value/BiasAdd/ReadVariableOp2F
!state_value/MatMul/ReadVariableOp!state_value/MatMul/ReadVariableOp:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
a
E__inference_Q_value_layer_call_and_return_conditional_losses_33059866

inputs
identity{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slicek
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims/dim?

ExpandDims
ExpandDimsstrided_slice:output:0ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2

ExpandDims
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice_1t
addAddV2ExpandDims:output:0strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2
add
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceinputsstrided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice_2r
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indices?
MeanMeanstrided_slice_2:output:0Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2
Mean[
subSubadd:z:0Mean:output:0*
T0*'
_output_shapes
:?????????2
sub[
IdentityIdentitysub:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
E__inference_Q_value_layer_call_and_return_conditional_losses_33059844

inputs
identity{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slicek
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims/dim?

ExpandDims
ExpandDimsstrided_slice:output:0ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2

ExpandDims
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice_1t
addAddV2ExpandDims:output:0strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2
add
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceinputsstrided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice_2r
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indices?
MeanMeanstrided_slice_2:output:0Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2
Mean[
subSubadd:z:0Mean:output:0*
T0*'
_output_shapes
:?????????2
sub[
IdentityIdentitysub:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
*__inference_dense_5_layer_call_fn_33059750

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_5_layer_call_and_return_conditional_losses_330591192
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?$
?
!__inference__traced_save_33059926
file_prefix-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop1
-savev2_state_value_kernel_read_readvariableop/
+savev2_state_value_bias_read_readvariableop/
+savev2_advantage_kernel_read_readvariableop-
)savev2_advantage_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
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
Const_1?
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
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop-savev2_state_value_kernel_read_readvariableop+savev2_state_value_bias_read_readvariableop+savev2_advantage_kernel_read_readvariableop)savev2_advantage_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
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

identity_1Identity_1:output:0*k
_input_shapesZ
X: ::::::::::: : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$	 

_output_shapes

:: 


_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?

?
I__inference_state_value_layer_call_and_return_conditional_losses_33059135

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
*__inference_model_1_layer_call_fn_33059568
inputs_0
inputs_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_model_1_layer_call_and_return_conditional_losses_330593732
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::?????????:?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
?
*__inference_dense_3_layer_call_fn_33059697

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_330590762
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?(
?
E__inference_model_1_layer_call_and_return_conditional_losses_33059488
series_data
latent_gain"
dense_3_33059459:
dense_3_33059461:"
dense_4_33059464:
dense_4_33059466:"
dense_5_33059470:
dense_5_33059472:&
state_value_33059475:"
state_value_33059477:$
advantage_33059480: 
advantage_33059482:
identity??!advantage/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?#state_value/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCallseries_datadense_3_33059459dense_3_33059461*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_330590762!
dense_3/StatefulPartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_33059464dense_4_33059466*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_4_layer_call_and_return_conditional_losses_330590932!
dense_4/StatefulPartitionedCall?
concatenate_2/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0latent_gain*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_concatenate_2_layer_call_and_return_conditional_losses_330591062
concatenate_2/PartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_5_33059470dense_5_33059472*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_5_layer_call_and_return_conditional_losses_330591192!
dense_5/StatefulPartitionedCall?
#state_value/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0state_value_33059475state_value_33059477*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_state_value_layer_call_and_return_conditional_losses_330591352%
#state_value/StatefulPartitionedCall?
!advantage/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0advantage_33059480advantage_33059482*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_advantage_layer_call_and_return_conditional_losses_330591512#
!advantage/StatefulPartitionedCall?
concatenate_3/PartitionedCallPartitionedCall,state_value/StatefulPartitionedCall:output:0*advantage/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_concatenate_3_layer_call_and_return_conditional_losses_330591642
concatenate_3/PartitionedCall?
Q_value/PartitionedCallPartitionedCall&concatenate_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_Q_value_layer_call_and_return_conditional_losses_330592442
Q_value/PartitionedCall{
IdentityIdentity Q_value/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp"^advantage/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall$^state_value/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::?????????:?????????: : : : : : : : : : 2F
!advantage/StatefulPartitionedCall!advantage/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2J
#state_value/StatefulPartitionedCall#state_value/StatefulPartitionedCall:T P
'
_output_shapes
:?????????
%
_user_specified_nameseries_data:TP
'
_output_shapes
:?????????
%
_user_specified_namelatent_gain
?
\
0__inference_concatenate_3_layer_call_fn_33059805
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_concatenate_3_layer_call_and_return_conditional_losses_330591642
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
?
E__inference_dense_3_layer_call_and_return_conditional_losses_33059708

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
I__inference_state_value_layer_call_and_return_conditional_losses_33059780

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
E__inference_dense_5_layer_call_and_return_conditional_losses_33059761

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
E__inference_Q_value_layer_call_and_return_conditional_losses_33059244

inputs
identity{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slicek
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims/dim?

ExpandDims
ExpandDimsstrided_slice:output:0ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2

ExpandDims
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice_1t
addAddV2ExpandDims:output:0strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2
add
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceinputsstrided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice_2r
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indices?
MeanMeanstrided_slice_2:output:0Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2
Mean[
subSubadd:z:0Mean:output:0*
T0*'
_output_shapes
:?????????2
sub[
IdentityIdentitysub:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
C
latent_gain4
serving_default_latent_gain:0?????????
C
series_data4
serving_default_series_data:0?????????;
Q_value0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer-8

layer-9
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
m__call__
*n&call_and_return_all_conditional_losses
o_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
?

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
p__call__
*q&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
r__call__
*s&call_and_return_all_conditional_losses"
_tf_keras_layer
"
_tf_keras_input_layer
?
	variables
regularization_losses
trainable_variables
 	keras_api
t__call__
*u&call_and_return_all_conditional_losses"
_tf_keras_layer
?

!kernel
"bias
#	variables
$regularization_losses
%trainable_variables
&	keras_api
v__call__
*w&call_and_return_all_conditional_losses"
_tf_keras_layer
?

'kernel
(bias
)	variables
*regularization_losses
+trainable_variables
,	keras_api
x__call__
*y&call_and_return_all_conditional_losses"
_tf_keras_layer
?

-kernel
.bias
/	variables
0regularization_losses
1trainable_variables
2	keras_api
z__call__
*{&call_and_return_all_conditional_losses"
_tf_keras_layer
?
3	variables
4regularization_losses
5trainable_variables
6	keras_api
|__call__
*}&call_and_return_all_conditional_losses"
_tf_keras_layer
?
7	variables
8regularization_losses
9trainable_variables
:	keras_api
~__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
"
	optimizer
f
0
1
2
3
!4
"5
'6
(7
-8
.9"
trackable_list_wrapper
 "
trackable_list_wrapper
f
0
1
2
3
!4
"5
'6
(7
-8
.9"
trackable_list_wrapper
?
	variables

;layers
<non_trainable_variables
=metrics
>layer_metrics
?layer_regularization_losses
regularization_losses
trainable_variables
m__call__
o_default_save_signature
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 :2dense_3/kernel
:2dense_3/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
	variables

@layers
Anon_trainable_variables
Bmetrics
Clayer_metrics
Dlayer_regularization_losses
regularization_losses
trainable_variables
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
 :2dense_4/kernel
:2dense_4/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
	variables

Elayers
Fnon_trainable_variables
Gmetrics
Hlayer_metrics
Ilayer_regularization_losses
regularization_losses
trainable_variables
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables

Jlayers
Knon_trainable_variables
Lmetrics
Mlayer_metrics
Nlayer_regularization_losses
regularization_losses
trainable_variables
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
 :2dense_5/kernel
:2dense_5/bias
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
?
#	variables

Olayers
Pnon_trainable_variables
Qmetrics
Rlayer_metrics
Slayer_regularization_losses
$regularization_losses
%trainable_variables
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
$:"2state_value/kernel
:2state_value/bias
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
?
)	variables

Tlayers
Unon_trainable_variables
Vmetrics
Wlayer_metrics
Xlayer_regularization_losses
*regularization_losses
+trainable_variables
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
": 2advantage/kernel
:2advantage/bias
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
?
/	variables

Ylayers
Znon_trainable_variables
[metrics
\layer_metrics
]layer_regularization_losses
0regularization_losses
1trainable_variables
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
3	variables

^layers
_non_trainable_variables
`metrics
alayer_metrics
blayer_regularization_losses
4regularization_losses
5trainable_variables
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
7	variables

clayers
dnon_trainable_variables
emetrics
flayer_metrics
glayer_regularization_losses
8regularization_losses
9trainable_variables
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
f
0
1
2
3
4
5
6
7
	8

9"
trackable_list_wrapper
 "
trackable_list_wrapper
'
h0"
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
N
	itotal
	jcount
k	variables
l	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
i0
j1"
trackable_list_wrapper
-
k	variables"
_generic_user_object
?2?
*__inference_model_1_layer_call_fn_33059214
*__inference_model_1_layer_call_fn_33059542
*__inference_model_1_layer_call_fn_33059568
*__inference_model_1_layer_call_fn_33059422?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_model_1_layer_call_and_return_conditional_losses_33059628
E__inference_model_1_layer_call_and_return_conditional_losses_33059688
E__inference_model_1_layer_call_and_return_conditional_losses_33059455
E__inference_model_1_layer_call_and_return_conditional_losses_33059488?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
#__inference__wrapped_model_33059056series_datalatent_gain"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_3_layer_call_fn_33059697?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_3_layer_call_and_return_conditional_losses_33059708?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_4_layer_call_fn_33059717?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_4_layer_call_and_return_conditional_losses_33059728?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_concatenate_2_layer_call_fn_33059734?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_concatenate_2_layer_call_and_return_conditional_losses_33059741?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_5_layer_call_fn_33059750?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_5_layer_call_and_return_conditional_losses_33059761?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_state_value_layer_call_fn_33059770?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_state_value_layer_call_and_return_conditional_losses_33059780?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_advantage_layer_call_fn_33059789?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_advantage_layer_call_and_return_conditional_losses_33059799?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_concatenate_3_layer_call_fn_33059805?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_concatenate_3_layer_call_and_return_conditional_losses_33059812?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_Q_value_layer_call_fn_33059817
*__inference_Q_value_layer_call_fn_33059822?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_Q_value_layer_call_and_return_conditional_losses_33059844
E__inference_Q_value_layer_call_and_return_conditional_losses_33059866?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
&__inference_signature_wrapper_33059516latent_gainseries_data"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
E__inference_Q_value_layer_call_and_return_conditional_losses_33059844`7?4
-?*
 ?
inputs?????????

 
p 
? "%?"
?
0?????????
? ?
E__inference_Q_value_layer_call_and_return_conditional_losses_33059866`7?4
-?*
 ?
inputs?????????

 
p
? "%?"
?
0?????????
? ?
*__inference_Q_value_layer_call_fn_33059817S7?4
-?*
 ?
inputs?????????

 
p 
? "???????????
*__inference_Q_value_layer_call_fn_33059822S7?4
-?*
 ?
inputs?????????

 
p
? "???????????
#__inference__wrapped_model_33059056?
!"'(-.`?]
V?S
Q?N
%?"
series_data?????????
%?"
latent_gain?????????
? "1?.
,
Q_value!?
Q_value??????????
G__inference_advantage_layer_call_and_return_conditional_losses_33059799\-./?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? 
,__inference_advantage_layer_call_fn_33059789O-./?,
%?"
 ?
inputs?????????
? "???????????
K__inference_concatenate_2_layer_call_and_return_conditional_losses_33059741?Z?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "%?"
?
0?????????
? ?
0__inference_concatenate_2_layer_call_fn_33059734vZ?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "???????????
K__inference_concatenate_3_layer_call_and_return_conditional_losses_33059812?Z?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "%?"
?
0?????????
? ?
0__inference_concatenate_3_layer_call_fn_33059805vZ?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "???????????
E__inference_dense_3_layer_call_and_return_conditional_losses_33059708\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? }
*__inference_dense_3_layer_call_fn_33059697O/?,
%?"
 ?
inputs?????????
? "???????????
E__inference_dense_4_layer_call_and_return_conditional_losses_33059728\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? }
*__inference_dense_4_layer_call_fn_33059717O/?,
%?"
 ?
inputs?????????
? "???????????
E__inference_dense_5_layer_call_and_return_conditional_losses_33059761\!"/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? }
*__inference_dense_5_layer_call_fn_33059750O!"/?,
%?"
 ?
inputs?????????
? "???????????
E__inference_model_1_layer_call_and_return_conditional_losses_33059455?
!"'(-.h?e
^?[
Q?N
%?"
series_data?????????
%?"
latent_gain?????????
p 

 
? "%?"
?
0?????????
? ?
E__inference_model_1_layer_call_and_return_conditional_losses_33059488?
!"'(-.h?e
^?[
Q?N
%?"
series_data?????????
%?"
latent_gain?????????
p

 
? "%?"
?
0?????????
? ?
E__inference_model_1_layer_call_and_return_conditional_losses_33059628?
!"'(-.b?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????
p 

 
? "%?"
?
0?????????
? ?
E__inference_model_1_layer_call_and_return_conditional_losses_33059688?
!"'(-.b?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????
p

 
? "%?"
?
0?????????
? ?
*__inference_model_1_layer_call_fn_33059214?
!"'(-.h?e
^?[
Q?N
%?"
series_data?????????
%?"
latent_gain?????????
p 

 
? "???????????
*__inference_model_1_layer_call_fn_33059422?
!"'(-.h?e
^?[
Q?N
%?"
series_data?????????
%?"
latent_gain?????????
p

 
? "???????????
*__inference_model_1_layer_call_fn_33059542?
!"'(-.b?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????
p 

 
? "???????????
*__inference_model_1_layer_call_fn_33059568?
!"'(-.b?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????
p

 
? "???????????
&__inference_signature_wrapper_33059516?
!"'(-.y?v
? 
o?l
4
latent_gain%?"
latent_gain?????????
4
series_data%?"
series_data?????????"1?.
,
Q_value!?
Q_value??????????
I__inference_state_value_layer_call_and_return_conditional_losses_33059780\'(/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? ?
.__inference_state_value_layer_call_fn_33059770O'(/?,
%?"
 ?
inputs?????????
? "??????????