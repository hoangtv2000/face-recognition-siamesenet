??
? ?
1
Acos
x"T
y"T"
Ttype:
2
	
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
h
Any	
input

reduction_indices"Tidx

output
"
	keep_dimsbool( "
Tidxtype0:
2	
?
ArgMax

input"T
	dimension"Tidx
output"output_type"!
Ttype:
2	
"
Tidxtype0:
2	"
output_typetype0	:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
,
Cos
x"T
y"T"
Ttype:

2
,
Exp
x"T
y"T"
Ttype:

2
?
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
+
IsNan
x"T
y
"
Ttype:
2
:
Less
x"T
y"T
z
"
Ttype:
2	
,
Log
x"T
y"T"
Ttype:

2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
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
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	?
0
Neg
x"T
y"T"
Ttype:
2
	
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
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
2
Round
x"T
y"T"
Ttype:
2
	
.
Rsqrt
x"T
y"T"
Ttype:

2
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
A
SelectV2
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
9
Softmax
logits"T
softmax"T"
Ttype:
2
7
Square
x"T
y"T"
Ttype:
2	
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
executor_typestring ??
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
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
f
TopKV2

input"T
k
values"T
indices"
sortedbool("
Ttype:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?
&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68ʺ
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
??*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:?*
dtype0
?
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_namebatch_normalization/gamma
?
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_namebatch_normalization/beta
?
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes	
:?*
dtype0
?
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*0
shared_name!batch_normalization/moving_mean
?
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes	
:?*
dtype0
?
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#batch_normalization/moving_variance
?
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes	
:?*
dtype0
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
??*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:?*
dtype0
?
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_1/gamma
?
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_namebatch_normalization_1/beta
?
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes	
:?*
dtype0
?
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!batch_normalization_1/moving_mean
?
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes	
:?*
dtype0
?
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*6
shared_name'%batch_normalization_1/moving_variance
?
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes	
:?*
dtype0
d
VariableVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable
]
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
: *
dtype0
p
	ada_cos/WVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_name	ada_cos/W
i
ada_cos/W/Read/ReadVariableOpReadVariableOp	ada_cos/W* 
_output_shapes
:
??*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
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
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*$
shared_nameAdam/dense/kernel/m
}
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m* 
_output_shapes
:
??*
dtype0
{
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/dense/bias/m
t
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes	
:?*
dtype0
?
 Adam/batch_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/batch_normalization/gamma/m
?
4Adam/batch_normalization/gamma/m/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/m*
_output_shapes	
:?*
dtype0
?
Adam/batch_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*0
shared_name!Adam/batch_normalization/beta/m
?
3Adam/batch_normalization/beta/m/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*&
shared_nameAdam/dense_1/kernel/m
?
)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m* 
_output_shapes
:
??*
dtype0

Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/dense_1/bias/m
x
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes	
:?*
dtype0
?
"Adam/batch_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_1/gamma/m
?
6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/m*
_output_shapes	
:?*
dtype0
?
!Adam/batch_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!Adam/batch_normalization_1/beta/m
?
5Adam/batch_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/m*
_output_shapes	
:?*
dtype0
~
Adam/ada_cos/W/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*!
shared_nameAdam/ada_cos/W/m
w
$Adam/ada_cos/W/m/Read/ReadVariableOpReadVariableOpAdam/ada_cos/W/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*$
shared_nameAdam/dense/kernel/v
}
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v* 
_output_shapes
:
??*
dtype0
{
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/dense/bias/v
t
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes	
:?*
dtype0
?
 Adam/batch_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/batch_normalization/gamma/v
?
4Adam/batch_normalization/gamma/v/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/v*
_output_shapes	
:?*
dtype0
?
Adam/batch_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*0
shared_name!Adam/batch_normalization/beta/v
?
3Adam/batch_normalization/beta/v/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*&
shared_nameAdam/dense_1/kernel/v
?
)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v* 
_output_shapes
:
??*
dtype0

Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/dense_1/bias/v
x
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes	
:?*
dtype0
?
"Adam/batch_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_1/gamma/v
?
6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/v*
_output_shapes	
:?*
dtype0
?
!Adam/batch_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!Adam/batch_normalization_1/beta/v
?
5Adam/batch_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/v*
_output_shapes	
:?*
dtype0
~
Adam/ada_cos/W/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*!
shared_nameAdam/ada_cos/W/v
w
$Adam/ada_cos/W/v/Read/ReadVariableOpReadVariableOpAdam/ada_cos/W/v* 
_output_shapes
:
??*
dtype0

NoOpNoOp
?J
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?J
value?JB?J B?I
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
		optimizer

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
?
axis
	gamma
beta
moving_mean
moving_variance
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses*
?
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)_random_generator
*__call__
*+&call_and_return_all_conditional_losses* 
?

,kernel
-bias
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses*
?
4axis
	5gamma
6beta
7moving_mean
8moving_variance
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses*
* 
?
?s
@W
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses*
?
Giter

Hbeta_1

Ibeta_2
	Jdecay
Klearning_ratem|m}m~m,m?-m?5m?6m?@m?v?v?v?v?,v?-v?5v?6v?@v?*
j
0
1
2
3
4
5
,6
-7
58
69
710
811
@12
?13*
C
0
1
2
3
,4
-5
56
67
@8*
	
L0* 
?
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics

	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

Rserving_default* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
?
Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
hb
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
0
1
2
3*

0
1*
* 
?
Xnon_trainable_variables

Ylayers
Zmetrics
[layer_regularization_losses
\layer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
]non_trainable_variables

^layers
_metrics
`layer_regularization_losses
alayer_metrics
%	variables
&trainable_variables
'regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses* 
* 
* 
* 
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

,0
-1*

,0
-1*
* 
?
bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses*
* 
* 
* 
jd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
50
61
72
83*

50
61*
* 
?
gnon_trainable_variables

hlayers
imetrics
jlayer_regularization_losses
klayer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses*
* 
* 
SM
VARIABLE_VALUEVariable1layer_with_weights-4/s/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUE	ada_cos/W1layer_with_weights-4/W/.ATTRIBUTES/VARIABLE_VALUE*

@0
?1*

@0*
	
L0* 
?
lnon_trainable_variables

mlayers
nmetrics
olayer_regularization_losses
player_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
'
0
1
72
83
?4*
<
0
1
2
3
4
5
6
7*

q0
r1*
* 
* 
* 
* 
* 
* 
* 
* 

0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

70
81*
* 
* 
* 
* 

?0*
* 
* 
	
L0* 
* 
8
	stotal
	tcount
u	variables
v	keras_api*
H
	wtotal
	xcount
y
_fn_kwargs
z	variables
{	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

s0
t1*

u	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

w0
x1*

z	variables*
y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE Adam/batch_normalization/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/batch_normalization/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_1/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE!Adam/batch_normalization_1/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUEAdam/ada_cos/W/mMlayer_with_weights-4/W/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE Adam/batch_normalization/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/batch_normalization/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_1/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE!Adam/batch_normalization_1/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUEAdam/ada_cos/W/vMlayer_with_weights-4/W/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?
serving_default_dense_inputPlaceholder*(
_output_shapes
:??????????*
dtype0*
shape:??????????
?
serving_default_onehot_labelPlaceholder*(
_output_shapes
:??????????*
dtype0*
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_dense_inputserving_default_onehot_labeldense/kernel
dense/bias#batch_normalization/moving_variancebatch_normalization/gammabatch_normalization/moving_meanbatch_normalization/betadense_1/kerneldense_1/bias%batch_normalization_1/moving_variancebatch_normalization_1/gamma!batch_normalization_1/moving_meanbatch_normalization_1/beta	ada_cos/WVariable*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference_signature_wrapper_285047
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOpVariable/Read/ReadVariableOpada_cos/W/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp4Adam/batch_normalization/gamma/m/Read/ReadVariableOp3Adam/batch_normalization/beta/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_1/beta/m/Read/ReadVariableOp$Adam/ada_cos/W/m/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp4Adam/batch_normalization/gamma/v/Read/ReadVariableOp3Adam/batch_normalization/beta/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_1/beta/v/Read/ReadVariableOp$Adam/ada_cos/W/v/Read/ReadVariableOpConst*6
Tin/
-2+	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *(
f#R!
__inference__traced_save_285553
?	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancedense_1/kerneldense_1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceVariable	ada_cos/W	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/dense/kernel/mAdam/dense/bias/m Adam/batch_normalization/gamma/mAdam/batch_normalization/beta/mAdam/dense_1/kernel/mAdam/dense_1/bias/m"Adam/batch_normalization_1/gamma/m!Adam/batch_normalization_1/beta/mAdam/ada_cos/W/mAdam/dense/kernel/vAdam/dense/bias/v Adam/batch_normalization/gamma/vAdam/batch_normalization/beta/vAdam/dense_1/kernel/vAdam/dense_1/bias/v"Adam/batch_normalization_1/gamma/v!Adam/batch_normalization_1/beta/vAdam/ada_cos/W/v*5
Tin.
,2**
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__traced_restore_285686??
?
?
&__inference_model_layer_call_fn_284301
dense_input
onehot_label
unknown:
??
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
	unknown_4:	?
	unknown_5:
??
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:	?

unknown_10:	?

unknown_11:
??

unknown_12: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_inputonehot_labelunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_284270p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:??????????:??????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
(
_output_shapes
:??????????
%
_user_specified_namedense_input:VR
(
_output_shapes
:??????????
&
_user_specified_nameonehot_label
?
?
$__inference_signature_wrapper_285047
dense_input
onehot_label
unknown:
??
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
	unknown_4:	?
	unknown_5:
??
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:	?

unknown_10:	?

unknown_11:
??

unknown_12: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_inputonehot_labelunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__wrapped_model_283920p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:??????????:??????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
(
_output_shapes
:??????????
%
_user_specified_namedense_input:VR
(
_output_shapes
:??????????
&
_user_specified_nameonehot_label
?,
?
A__inference_model_layer_call_and_return_conditional_losses_284592
dense_input
onehot_label 
dense_284551:
??
dense_284553:	?)
batch_normalization_284556:	?)
batch_normalization_284558:	?)
batch_normalization_284560:	?)
batch_normalization_284562:	?"
dense_1_284566:
??
dense_1_284568:	?+
batch_normalization_1_284571:	?+
batch_normalization_1_284573:	?+
batch_normalization_1_284575:	?+
batch_normalization_1_284577:	?"
ada_cos_284580:
??
ada_cos_284582: 
identity??ada_cos/StatefulPartitionedCall?+ada_cos/W/Regularizer/Square/ReadVariableOp?+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dropout/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCalldense_inputdense_284551dense_284553*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_284104?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_284556batch_normalization_284558batch_normalization_284560batch_normalization_284562*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_283991?
dropout/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_284342?
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_284566dense_1_284568*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_284137?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_1_284571batch_normalization_1_284573batch_normalization_1_284575batch_normalization_1_284577*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_284073?
ada_cos/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0onehot_labelada_cos_284580ada_cos_284582*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_ada_cos_layer_call_and_return_conditional_losses_284257|
+ada_cos/W/Regularizer/Square/ReadVariableOpReadVariableOpada_cos_284580* 
_output_shapes
:
??*
dtype0?
ada_cos/W/Regularizer/SquareSquare3ada_cos/W/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??l
ada_cos/W/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
ada_cos/W/Regularizer/SumSum ada_cos/W/Regularizer/Square:y:0$ada_cos/W/Regularizer/Const:output:0*
T0*
_output_shapes
: `
ada_cos/W/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
ada_cos/W/Regularizer/mulMul$ada_cos/W/Regularizer/mul/x:output:0"ada_cos/W/Regularizer/Sum:output:0*
T0*
_output_shapes
: x
IdentityIdentity(ada_cos/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp ^ada_cos/StatefulPartitionedCall,^ada_cos/W/Regularizer/Square/ReadVariableOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:??????????:??????????: : : : : : : : : : : : : : 2B
ada_cos/StatefulPartitionedCallada_cos/StatefulPartitionedCall2Z
+ada_cos/W/Regularizer/Square/ReadVariableOp+ada_cos/W/Regularizer/Square/ReadVariableOp2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall:U Q
(
_output_shapes
:??????????
%
_user_specified_namedense_input:VR
(
_output_shapes
:??????????
&
_user_specified_nameonehot_label
?%
?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_285147

inputs6
'assignmovingavg_readvariableop_resource:	?8
)assignmovingavg_1_readvariableop_resource:	?4
%batchnorm_mul_readvariableop_resource:	?0
!batchnorm_readvariableop_resource:	?
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	??
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:??????????l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:?y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:??
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:??
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?%
?
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_285274

inputs6
'assignmovingavg_readvariableop_resource:	?8
)assignmovingavg_1_readvariableop_resource:	?4
%batchnorm_mul_readvariableop_resource:	?0
!batchnorm_readvariableop_resource:	?
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	??
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:??????????l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:?y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:??
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:??
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?%
?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_283991

inputs6
'assignmovingavg_readvariableop_resource:	?8
)assignmovingavg_1_readvariableop_resource:	?4
%batchnorm_mul_readvariableop_resource:	?0
!batchnorm_readvariableop_resource:	?
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	??
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:??????????l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:?y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:??
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:??
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?a
?
C__inference_ada_cos_layer_call_and_return_conditional_losses_285395
inputs_0
inputs_1:
&l2_normalize_1_readvariableop_resource:
??!
readvariableop_resource: 
identity??AssignVariableOp?ReadVariableOp?ReadVariableOp_1?+ada_cos/W/Regularizer/Square/ReadVariableOp?l2_normalize_1/ReadVariableOpZ
l2_normalize/SquareSquareinputs_0*
T0*(
_output_shapes
:??????????d
"l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
l2_normalize/SumSuml2_normalize/Square:y:0+l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims([
l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼?+?
l2_normalize/MaximumMaximuml2_normalize/Sum:output:0l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:?????????g
l2_normalize/RsqrtRsqrtl2_normalize/Maximum:z:0*
T0*'
_output_shapes
:?????????h
l2_normalizeMulinputs_0l2_normalize/Rsqrt:y:0*
T0*(
_output_shapes
:???????????
l2_normalize_1/ReadVariableOpReadVariableOp&l2_normalize_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0q
l2_normalize_1/SquareSquare%l2_normalize_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??f
$l2_normalize_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : ?
l2_normalize_1/SumSuml2_normalize_1/Square:y:0-l2_normalize_1/Sum/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(]
l2_normalize_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼?+?
l2_normalize_1/MaximumMaximuml2_normalize_1/Sum:output:0!l2_normalize_1/Maximum/y:output:0*
T0*
_output_shapes
:	?c
l2_normalize_1/RsqrtRsqrtl2_normalize_1/Maximum:z:0*
T0*
_output_shapes
:	??
l2_normalize_1Mul%l2_normalize_1/ReadVariableOp:value:0l2_normalize_1/Rsqrt:y:0*
T0* 
_output_shapes
:
??i
matmulMatMull2_normalize:z:0l2_normalize_1:z:0*
T0*(
_output_shapes
:??????????\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *????
clip_by_value/MinimumMinimummatmul:product:0 clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *????
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*(
_output_shapes
:??????????R
AcosAcosclip_by_value:z:0*
T0*(
_output_shapes
:??????????H
Less/yConst*
_output_shapes
: *
dtype0*
value	B :Z
LessLessinputs_1Less/y:output:0*
T0*(
_output_shapes
:??????????^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0g
mulMulReadVariableOp:value:0matmul:product:0*
T0*(
_output_shapes
:??????????F
ExpExpmul:z:0*
T0*(
_output_shapes
:??????????\

zeros_like	ZerosLikematmul:product:0*
T0*(
_output_shapes
:??????????j
SelectV2SelectV2Less:z:0Exp:y:0zeros_like:y:0*
T0*(
_output_shapes
:??????????W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :k
SumSumSelectV2:output:0Sum/reduction_indices:output:0*
T0*#
_output_shapes
:?????????O
ConstConst*
_output_shapes
:*
dtype0*
valueB: L
B_avgMeanSum:output:0Const:output:0*
T0*
_output_shapes
: R
ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
value	B :c
ArgMaxArgMaxinputs_1ArgMax/dimension:output:0*
T0*#
_output_shapes
:?????????R
theta_class/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
theta_classGatherV2Acos:y:0ArgMax:output:0theta_class/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*(
_output_shapes
:??????????S
percentile/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2c
percentile/CastCastpercentile/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: k
percentile/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
percentile/ReshapeReshapetheta_class:output:0!percentile/Reshape/shape:output:0*
T0*#
_output_shapes
:?????????]
percentile/truediv/yConst*
_output_shapes
: *
dtype0*
valueB 2      Y@r
percentile/truedivRealDivpercentile/Cast:y:0percentile/truediv/y:output:0*
T0*
_output_shapes
: _
percentile/sort/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????e
percentile/sort/NegNegpercentile/Reshape:output:0*
T0*#
_output_shapes
:?????????\
percentile/sort/ShapeShapepercentile/sort/Neg:y:0*
T0*
_output_shapes
:v
#percentile/sort/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????o
%percentile/sort/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%percentile/sort/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
percentile/sort/strided_sliceStridedSlicepercentile/sort/Shape:output:0,percentile/sort/strided_slice/stack:output:0.percentile/sort/strided_slice/stack_1:output:0.percentile/sort/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
percentile/sort/RankConst*
_output_shapes
: *
dtype0*
value	B :?
percentile/sort/TopKV2TopKV2percentile/sort/Neg:y:0&percentile/sort/strided_slice:output:0*
T0*2
_output_shapes 
:?????????:?????????k
percentile/sort/Neg_1Negpercentile/sort/TopKV2:values:0*
T0*#
_output_shapes
:?????????[
percentile/ShapeShapepercentile/Reshape:output:0*
T0*
_output_shapes
:q
percentile/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????j
 percentile/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: j
 percentile/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
percentile/strided_sliceStridedSlicepercentile/Shape:output:0'percentile/strided_slice/stack:output:0)percentile/strided_slice/stack_1:output:0)percentile/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
percentile/Cast_1Cast!percentile/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: Y
percentile/sub/yConst*
_output_shapes
: *
dtype0*
valueB 2      ??h
percentile/subSubpercentile/Cast_1:y:0percentile/sub/y:output:0*
T0*
_output_shapes
: b
percentile/mulMulpercentile/sub:z:0percentile/truediv:z:0*
T0*
_output_shapes
: N
percentile/RoundRoundpercentile/mul:z:0*
T0*
_output_shapes
: _
percentile/Cast_2Castpercentile/Round:y:0*

DstT0*

SrcT0*
_output_shapes
: ]
percentile/Shape_1Shapepercentile/Reshape:output:0*
T0*
_output_shapes
:s
 percentile/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????l
"percentile/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: l
"percentile/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
percentile/strided_slice_1StridedSlicepercentile/Shape_1:output:0)percentile/strided_slice_1/stack:output:0+percentile/strided_slice_1/stack_1:output:0+percentile/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
percentile/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :z
percentile/sub_1Sub#percentile/strided_slice_1:output:0percentile/sub_1/y:output:0*
T0*
_output_shapes
: y
 percentile/clip_by_value/MinimumMinimumpercentile/Cast_2:y:0percentile/sub_1:z:0*
T0*
_output_shapes
: \
percentile/clip_by_value/yConst*
_output_shapes
: *
dtype0*
value	B : ?
percentile/clip_by_valueMaximum$percentile/clip_by_value/Minimum:z:0#percentile/clip_by_value/y:output:0*
T0*
_output_shapes
: c
percentile/GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
percentile/GatherV2GatherV2percentile/sort/Neg_1:y:0percentile/clip_by_value:z:0!percentile/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: b
percentile/IsNanIsNantheta_class:output:0*
T0*(
_output_shapes
:??????????a
percentile/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ^
percentile/AnyAnypercentile/IsNan:y:0percentile/Const:output:0*
_output_shapes
: ]
percentile/Reshape_1/shapeConst*
_output_shapes
: *
dtype0*
valueB ~
percentile/Reshape_1Reshapepercentile/Any:output:0#percentile/Reshape_1/shape:output:0*
T0
*
_output_shapes
: Z
percentile/SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *  ??
percentile/SelectV2SelectV2percentile/Reshape_1:output:0percentile/SelectV2/t:output:0percentile/GatherV2:output:0*
T0*
_output_shapes
: c
!percentile/rotate_transpose/shiftConst*
_output_shapes
: *
dtype0*
value	B : j
Lpercentile/rotate_transpose/assert_integer/statically_determined_was_integerNoOp*
_output_shapes
 Q
LogLogB_avg:output:0^percentile/SelectV2*
T0*
_output_shapes
: l
	Minimum/xConst^B_avg^percentile/SelectV2*
_output_shapes
: *
dtype0*
valueB
 *?I?e
MinimumMinimumMinimum/x:output:0percentile/SelectV2:output:0*
T0*
_output_shapes
: 8
CosCosMinimum:z:0*
T0*
_output_shapes
: E
truedivRealDivLog:y:0Cos:y:0*
T0*
_output_shapes
: ~
AssignVariableOpAssignVariableOpreadvariableop_resourcetruediv:z:0^ReadVariableOp*
_output_shapes
 *
dtype0?
ReadVariableOp_1ReadVariableOpreadvariableop_resource^AssignVariableOp^B_avg^percentile/SelectV2*
_output_shapes
: *
dtype0k
mul_1MulReadVariableOp_1:value:0matmul:product:0*
T0*(
_output_shapes
:??????????P
SoftmaxSoftmax	mul_1:z:0*
T0*(
_output_shapes
:???????????
+ada_cos/W/Regularizer/Square/ReadVariableOpReadVariableOp&l2_normalize_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
ada_cos/W/Regularizer/SquareSquare3ada_cos/W/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??l
ada_cos/W/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
ada_cos/W/Regularizer/SumSum ada_cos/W/Regularizer/Square:y:0$ada_cos/W/Regularizer/Const:output:0*
T0*
_output_shapes
: `
ada_cos/W/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
ada_cos/W/Regularizer/mulMul$ada_cos/W/Regularizer/mul/x:output:0"ada_cos/W/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^AssignVariableOp^ReadVariableOp^ReadVariableOp_1,^ada_cos/W/Regularizer/Square/ReadVariableOp^l2_normalize_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:??????????:??????????: : 2$
AssignVariableOpAssignVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12Z
+ada_cos/W/Regularizer/Square/ReadVariableOp+ada_cos/W/Regularizer/Square/ReadVariableOp2>
l2_normalize_1/ReadVariableOpl2_normalize_1/ReadVariableOp:R N
(
_output_shapes
:??????????
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/1
?U
?
__inference__traced_save_285553
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop'
#savev2_variable_read_readvariableop(
$savev2_ada_cos_w_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop?
;savev2_adam_batch_normalization_gamma_m_read_readvariableop>
:savev2_adam_batch_normalization_beta_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_1_beta_m_read_readvariableop/
+savev2_adam_ada_cos_w_m_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop?
;savev2_adam_batch_normalization_gamma_v_read_readvariableop>
:savev2_adam_batch_normalization_beta_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_1_beta_v_read_readvariableop/
+savev2_adam_ada_cos_w_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:**
dtype0*?
value?B?*B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-4/s/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-4/W/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-4/W/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-4/W/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:**
dtype0*g
value^B\*B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop#savev2_variable_read_readvariableop$savev2_ada_cos_w_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop;savev2_adam_batch_normalization_gamma_m_read_readvariableop:savev2_adam_batch_normalization_beta_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop<savev2_adam_batch_normalization_1_beta_m_read_readvariableop+savev2_adam_ada_cos_w_m_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop;savev2_adam_batch_normalization_gamma_v_read_readvariableop:savev2_adam_batch_normalization_beta_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop<savev2_adam_batch_normalization_1_beta_v_read_readvariableop+savev2_adam_ada_cos_w_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *8
dtypes.
,2*	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :
??:?:?:?:?:?:
??:?:?:?:?:?: :
??: : : : : : : : : :
??:?:?:?:
??:?:?:?:
??:
??:?:?:?:
??:?:?:?:
??: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:!	

_output_shapes	
:?:!


_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:

_output_shapes
: :&"
 
_output_shapes
:
??:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
??:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:& "
 
_output_shapes
:
??:&!"
 
_output_shapes
:
??:!"

_output_shapes	
:?:!#

_output_shapes	
:?:!$

_output_shapes	
:?:&%"
 
_output_shapes
:
??:!&

_output_shapes	
:?:!'

_output_shapes	
:?:!(

_output_shapes	
:?:&)"
 
_output_shapes
:
??:*

_output_shapes
: 
?
?
(__inference_ada_cos_layer_call_fn_285290
inputs_0
inputs_1
unknown:
??
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_ada_cos_layer_call_and_return_conditional_losses_284257p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:??????????:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:??????????
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/1
?
a
(__inference_dropout_layer_call_fn_285157

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_284342p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_1_layer_call_fn_285207

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_284026p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
A__inference_dense_layer_call_and_return_conditional_losses_285067

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?a
?
C__inference_ada_cos_layer_call_and_return_conditional_losses_284257

inputs
inputs_1:
&l2_normalize_1_readvariableop_resource:
??!
readvariableop_resource: 
identity??AssignVariableOp?ReadVariableOp?ReadVariableOp_1?+ada_cos/W/Regularizer/Square/ReadVariableOp?l2_normalize_1/ReadVariableOpX
l2_normalize/SquareSquareinputs*
T0*(
_output_shapes
:??????????d
"l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
l2_normalize/SumSuml2_normalize/Square:y:0+l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims([
l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼?+?
l2_normalize/MaximumMaximuml2_normalize/Sum:output:0l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:?????????g
l2_normalize/RsqrtRsqrtl2_normalize/Maximum:z:0*
T0*'
_output_shapes
:?????????f
l2_normalizeMulinputsl2_normalize/Rsqrt:y:0*
T0*(
_output_shapes
:???????????
l2_normalize_1/ReadVariableOpReadVariableOp&l2_normalize_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0q
l2_normalize_1/SquareSquare%l2_normalize_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??f
$l2_normalize_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : ?
l2_normalize_1/SumSuml2_normalize_1/Square:y:0-l2_normalize_1/Sum/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(]
l2_normalize_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼?+?
l2_normalize_1/MaximumMaximuml2_normalize_1/Sum:output:0!l2_normalize_1/Maximum/y:output:0*
T0*
_output_shapes
:	?c
l2_normalize_1/RsqrtRsqrtl2_normalize_1/Maximum:z:0*
T0*
_output_shapes
:	??
l2_normalize_1Mul%l2_normalize_1/ReadVariableOp:value:0l2_normalize_1/Rsqrt:y:0*
T0* 
_output_shapes
:
??i
matmulMatMull2_normalize:z:0l2_normalize_1:z:0*
T0*(
_output_shapes
:??????????\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *????
clip_by_value/MinimumMinimummatmul:product:0 clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *????
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*(
_output_shapes
:??????????R
AcosAcosclip_by_value:z:0*
T0*(
_output_shapes
:??????????H
Less/yConst*
_output_shapes
: *
dtype0*
value	B :Z
LessLessinputs_1Less/y:output:0*
T0*(
_output_shapes
:??????????^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0g
mulMulReadVariableOp:value:0matmul:product:0*
T0*(
_output_shapes
:??????????F
ExpExpmul:z:0*
T0*(
_output_shapes
:??????????\

zeros_like	ZerosLikematmul:product:0*
T0*(
_output_shapes
:??????????j
SelectV2SelectV2Less:z:0Exp:y:0zeros_like:y:0*
T0*(
_output_shapes
:??????????W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :k
SumSumSelectV2:output:0Sum/reduction_indices:output:0*
T0*#
_output_shapes
:?????????O
ConstConst*
_output_shapes
:*
dtype0*
valueB: L
B_avgMeanSum:output:0Const:output:0*
T0*
_output_shapes
: R
ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
value	B :c
ArgMaxArgMaxinputs_1ArgMax/dimension:output:0*
T0*#
_output_shapes
:?????????R
theta_class/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
theta_classGatherV2Acos:y:0ArgMax:output:0theta_class/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*(
_output_shapes
:??????????S
percentile/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2c
percentile/CastCastpercentile/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: k
percentile/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
percentile/ReshapeReshapetheta_class:output:0!percentile/Reshape/shape:output:0*
T0*#
_output_shapes
:?????????]
percentile/truediv/yConst*
_output_shapes
: *
dtype0*
valueB 2      Y@r
percentile/truedivRealDivpercentile/Cast:y:0percentile/truediv/y:output:0*
T0*
_output_shapes
: _
percentile/sort/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????e
percentile/sort/NegNegpercentile/Reshape:output:0*
T0*#
_output_shapes
:?????????\
percentile/sort/ShapeShapepercentile/sort/Neg:y:0*
T0*
_output_shapes
:v
#percentile/sort/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????o
%percentile/sort/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%percentile/sort/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
percentile/sort/strided_sliceStridedSlicepercentile/sort/Shape:output:0,percentile/sort/strided_slice/stack:output:0.percentile/sort/strided_slice/stack_1:output:0.percentile/sort/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
percentile/sort/RankConst*
_output_shapes
: *
dtype0*
value	B :?
percentile/sort/TopKV2TopKV2percentile/sort/Neg:y:0&percentile/sort/strided_slice:output:0*
T0*2
_output_shapes 
:?????????:?????????k
percentile/sort/Neg_1Negpercentile/sort/TopKV2:values:0*
T0*#
_output_shapes
:?????????[
percentile/ShapeShapepercentile/Reshape:output:0*
T0*
_output_shapes
:q
percentile/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????j
 percentile/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: j
 percentile/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
percentile/strided_sliceStridedSlicepercentile/Shape:output:0'percentile/strided_slice/stack:output:0)percentile/strided_slice/stack_1:output:0)percentile/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
percentile/Cast_1Cast!percentile/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: Y
percentile/sub/yConst*
_output_shapes
: *
dtype0*
valueB 2      ??h
percentile/subSubpercentile/Cast_1:y:0percentile/sub/y:output:0*
T0*
_output_shapes
: b
percentile/mulMulpercentile/sub:z:0percentile/truediv:z:0*
T0*
_output_shapes
: N
percentile/RoundRoundpercentile/mul:z:0*
T0*
_output_shapes
: _
percentile/Cast_2Castpercentile/Round:y:0*

DstT0*

SrcT0*
_output_shapes
: ]
percentile/Shape_1Shapepercentile/Reshape:output:0*
T0*
_output_shapes
:s
 percentile/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????l
"percentile/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: l
"percentile/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
percentile/strided_slice_1StridedSlicepercentile/Shape_1:output:0)percentile/strided_slice_1/stack:output:0+percentile/strided_slice_1/stack_1:output:0+percentile/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
percentile/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :z
percentile/sub_1Sub#percentile/strided_slice_1:output:0percentile/sub_1/y:output:0*
T0*
_output_shapes
: y
 percentile/clip_by_value/MinimumMinimumpercentile/Cast_2:y:0percentile/sub_1:z:0*
T0*
_output_shapes
: \
percentile/clip_by_value/yConst*
_output_shapes
: *
dtype0*
value	B : ?
percentile/clip_by_valueMaximum$percentile/clip_by_value/Minimum:z:0#percentile/clip_by_value/y:output:0*
T0*
_output_shapes
: c
percentile/GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
percentile/GatherV2GatherV2percentile/sort/Neg_1:y:0percentile/clip_by_value:z:0!percentile/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: b
percentile/IsNanIsNantheta_class:output:0*
T0*(
_output_shapes
:??????????a
percentile/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ^
percentile/AnyAnypercentile/IsNan:y:0percentile/Const:output:0*
_output_shapes
: ]
percentile/Reshape_1/shapeConst*
_output_shapes
: *
dtype0*
valueB ~
percentile/Reshape_1Reshapepercentile/Any:output:0#percentile/Reshape_1/shape:output:0*
T0
*
_output_shapes
: Z
percentile/SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *  ??
percentile/SelectV2SelectV2percentile/Reshape_1:output:0percentile/SelectV2/t:output:0percentile/GatherV2:output:0*
T0*
_output_shapes
: c
!percentile/rotate_transpose/shiftConst*
_output_shapes
: *
dtype0*
value	B : j
Lpercentile/rotate_transpose/assert_integer/statically_determined_was_integerNoOp*
_output_shapes
 Q
LogLogB_avg:output:0^percentile/SelectV2*
T0*
_output_shapes
: l
	Minimum/xConst^B_avg^percentile/SelectV2*
_output_shapes
: *
dtype0*
valueB
 *?I?e
MinimumMinimumMinimum/x:output:0percentile/SelectV2:output:0*
T0*
_output_shapes
: 8
CosCosMinimum:z:0*
T0*
_output_shapes
: E
truedivRealDivLog:y:0Cos:y:0*
T0*
_output_shapes
: ~
AssignVariableOpAssignVariableOpreadvariableop_resourcetruediv:z:0^ReadVariableOp*
_output_shapes
 *
dtype0?
ReadVariableOp_1ReadVariableOpreadvariableop_resource^AssignVariableOp^B_avg^percentile/SelectV2*
_output_shapes
: *
dtype0k
mul_1MulReadVariableOp_1:value:0matmul:product:0*
T0*(
_output_shapes
:??????????P
SoftmaxSoftmax	mul_1:z:0*
T0*(
_output_shapes
:???????????
+ada_cos/W/Regularizer/Square/ReadVariableOpReadVariableOp&l2_normalize_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
ada_cos/W/Regularizer/SquareSquare3ada_cos/W/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??l
ada_cos/W/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
ada_cos/W/Regularizer/SumSum ada_cos/W/Regularizer/Square:y:0$ada_cos/W/Regularizer/Const:output:0*
T0*
_output_shapes
: `
ada_cos/W/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
ada_cos/W/Regularizer/mulMul$ada_cos/W/Regularizer/mul/x:output:0"ada_cos/W/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^AssignVariableOp^ReadVariableOp^ReadVariableOp_1,^ada_cos/W/Regularizer/Square/ReadVariableOp^l2_normalize_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:??????????:??????????: : 2$
AssignVariableOpAssignVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12Z
+ada_cos/W/Regularizer/Square/ReadVariableOp+ada_cos/W/Regularizer/Square/ReadVariableOp2>
l2_normalize_1/ReadVariableOpl2_normalize_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
a
C__inference_dropout_layer_call_and_return_conditional_losses_284124

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
C__inference_dense_1_layer_call_and_return_conditional_losses_285194

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
&__inference_model_layer_call_fn_284672
inputs_0
inputs_1
unknown:
??
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
	unknown_4:	?
	unknown_5:
??
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:	?

unknown_10:	?

unknown_11:
??

unknown_12: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*+
_read_only_resource_inputs
		*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_284437p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:??????????:??????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:??????????
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/1
?
?
&__inference_model_layer_call_fn_284638
inputs_0
inputs_1
unknown:
??
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
	unknown_4:	?
	unknown_5:
??
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:	?

unknown_10:	?

unknown_11:
??

unknown_12: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_284270p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:??????????:??????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:??????????
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/1
?
a
C__inference_dropout_layer_call_and_return_conditional_losses_285162

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_285113

inputs0
!batchnorm_readvariableop_resource:	?4
%batchnorm_mul_readvariableop_resource:	?2
#batchnorm_readvariableop_1_resource:	?2
#batchnorm_readvariableop_2_resource:	?
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
!__inference__wrapped_model_283920
dense_input
onehot_label>
*model_dense_matmul_readvariableop_resource:
??:
+model_dense_biasadd_readvariableop_resource:	?J
;model_batch_normalization_batchnorm_readvariableop_resource:	?N
?model_batch_normalization_batchnorm_mul_readvariableop_resource:	?L
=model_batch_normalization_batchnorm_readvariableop_1_resource:	?L
=model_batch_normalization_batchnorm_readvariableop_2_resource:	?@
,model_dense_1_matmul_readvariableop_resource:
??<
-model_dense_1_biasadd_readvariableop_resource:	?L
=model_batch_normalization_1_batchnorm_readvariableop_resource:	?P
Amodel_batch_normalization_1_batchnorm_mul_readvariableop_resource:	?N
?model_batch_normalization_1_batchnorm_readvariableop_1_resource:	?N
?model_batch_normalization_1_batchnorm_readvariableop_2_resource:	?H
4model_ada_cos_l2_normalize_1_readvariableop_resource:
??/
%model_ada_cos_readvariableop_resource: 
identity??model/ada_cos/AssignVariableOp?model/ada_cos/ReadVariableOp?model/ada_cos/ReadVariableOp_1?+model/ada_cos/l2_normalize_1/ReadVariableOp?2model/batch_normalization/batchnorm/ReadVariableOp?4model/batch_normalization/batchnorm/ReadVariableOp_1?4model/batch_normalization/batchnorm/ReadVariableOp_2?6model/batch_normalization/batchnorm/mul/ReadVariableOp?4model/batch_normalization_1/batchnorm/ReadVariableOp?6model/batch_normalization_1/batchnorm/ReadVariableOp_1?6model/batch_normalization_1/batchnorm/ReadVariableOp_2?8model/batch_normalization_1/batchnorm/mul/ReadVariableOp?"model/dense/BiasAdd/ReadVariableOp?!model/dense/MatMul/ReadVariableOp?$model/dense_1/BiasAdd/ReadVariableOp?#model/dense_1/MatMul/ReadVariableOp?
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
model/dense/MatMulMatMuldense_input)model/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????i
model/dense/ReluRelumodel/dense/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
2model/batch_normalization/batchnorm/ReadVariableOpReadVariableOp;model_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype0n
)model/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
'model/batch_normalization/batchnorm/addAddV2:model/batch_normalization/batchnorm/ReadVariableOp:value:02model/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:??
)model/batch_normalization/batchnorm/RsqrtRsqrt+model/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:??
6model/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp?model_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'model/batch_normalization/batchnorm/mulMul-model/batch_normalization/batchnorm/Rsqrt:y:0>model/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:??
)model/batch_normalization/batchnorm/mul_1Mulmodel/dense/Relu:activations:0+model/batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:???????????
4model/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp=model_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
)model/batch_normalization/batchnorm/mul_2Mul<model/batch_normalization/batchnorm/ReadVariableOp_1:value:0+model/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:??
4model/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp=model_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype0?
'model/batch_normalization/batchnorm/subSub<model/batch_normalization/batchnorm/ReadVariableOp_2:value:0-model/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:??
)model/batch_normalization/batchnorm/add_1AddV2-model/batch_normalization/batchnorm/mul_1:z:0+model/batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:???????????
model/dropout/IdentityIdentity-model/batch_normalization/batchnorm/add_1:z:0*
T0*(
_output_shapes
:???????????
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
model/dense_1/MatMulMatMulmodel/dropout/Identity:output:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????m
model/dense_1/ReluRelumodel/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
4model/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp=model_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype0p
+model/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
)model/batch_normalization_1/batchnorm/addAddV2<model/batch_normalization_1/batchnorm/ReadVariableOp:value:04model/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:??
+model/batch_normalization_1/batchnorm/RsqrtRsqrt-model/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:??
8model/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpAmodel_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype0?
)model/batch_normalization_1/batchnorm/mulMul/model/batch_normalization_1/batchnorm/Rsqrt:y:0@model/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:??
+model/batch_normalization_1/batchnorm/mul_1Mul model/dense_1/Relu:activations:0-model/batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:???????????
6model/batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOp?model_batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
+model/batch_normalization_1/batchnorm/mul_2Mul>model/batch_normalization_1/batchnorm/ReadVariableOp_1:value:0-model/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:??
6model/batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOp?model_batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype0?
)model/batch_normalization_1/batchnorm/subSub>model/batch_normalization_1/batchnorm/ReadVariableOp_2:value:0/model/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:??
+model/batch_normalization_1/batchnorm/add_1AddV2/model/batch_normalization_1/batchnorm/mul_1:z:0-model/batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:???????????
!model/ada_cos/l2_normalize/SquareSquare/model/batch_normalization_1/batchnorm/add_1:z:0*
T0*(
_output_shapes
:??????????r
0model/ada_cos/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
model/ada_cos/l2_normalize/SumSum%model/ada_cos/l2_normalize/Square:y:09model/ada_cos/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(i
$model/ada_cos/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼?+?
"model/ada_cos/l2_normalize/MaximumMaximum'model/ada_cos/l2_normalize/Sum:output:0-model/ada_cos/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:??????????
 model/ada_cos/l2_normalize/RsqrtRsqrt&model/ada_cos/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:??????????
model/ada_cos/l2_normalizeMul/model/batch_normalization_1/batchnorm/add_1:z:0$model/ada_cos/l2_normalize/Rsqrt:y:0*
T0*(
_output_shapes
:???????????
+model/ada_cos/l2_normalize_1/ReadVariableOpReadVariableOp4model_ada_cos_l2_normalize_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
#model/ada_cos/l2_normalize_1/SquareSquare3model/ada_cos/l2_normalize_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??t
2model/ada_cos/l2_normalize_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : ?
 model/ada_cos/l2_normalize_1/SumSum'model/ada_cos/l2_normalize_1/Square:y:0;model/ada_cos/l2_normalize_1/Sum/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(k
&model/ada_cos/l2_normalize_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼?+?
$model/ada_cos/l2_normalize_1/MaximumMaximum)model/ada_cos/l2_normalize_1/Sum:output:0/model/ada_cos/l2_normalize_1/Maximum/y:output:0*
T0*
_output_shapes
:	?
"model/ada_cos/l2_normalize_1/RsqrtRsqrt(model/ada_cos/l2_normalize_1/Maximum:z:0*
T0*
_output_shapes
:	??
model/ada_cos/l2_normalize_1Mul3model/ada_cos/l2_normalize_1/ReadVariableOp:value:0&model/ada_cos/l2_normalize_1/Rsqrt:y:0*
T0* 
_output_shapes
:
???
model/ada_cos/matmulMatMulmodel/ada_cos/l2_normalize:z:0 model/ada_cos/l2_normalize_1:z:0*
T0*(
_output_shapes
:??????????j
%model/ada_cos/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *????
#model/ada_cos/clip_by_value/MinimumMinimummodel/ada_cos/matmul:product:0.model/ada_cos/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????b
model/ada_cos/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *????
model/ada_cos/clip_by_valueMaximum'model/ada_cos/clip_by_value/Minimum:z:0&model/ada_cos/clip_by_value/y:output:0*
T0*(
_output_shapes
:??????????n
model/ada_cos/AcosAcosmodel/ada_cos/clip_by_value:z:0*
T0*(
_output_shapes
:??????????V
model/ada_cos/Less/yConst*
_output_shapes
: *
dtype0*
value	B :z
model/ada_cos/LessLessonehot_labelmodel/ada_cos/Less/y:output:0*
T0*(
_output_shapes
:??????????z
model/ada_cos/ReadVariableOpReadVariableOp%model_ada_cos_readvariableop_resource*
_output_shapes
: *
dtype0?
model/ada_cos/mulMul$model/ada_cos/ReadVariableOp:value:0model/ada_cos/matmul:product:0*
T0*(
_output_shapes
:??????????b
model/ada_cos/ExpExpmodel/ada_cos/mul:z:0*
T0*(
_output_shapes
:??????????x
model/ada_cos/zeros_like	ZerosLikemodel/ada_cos/matmul:product:0*
T0*(
_output_shapes
:???????????
model/ada_cos/SelectV2SelectV2model/ada_cos/Less:z:0model/ada_cos/Exp:y:0model/ada_cos/zeros_like:y:0*
T0*(
_output_shapes
:??????????e
#model/ada_cos/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
model/ada_cos/SumSummodel/ada_cos/SelectV2:output:0,model/ada_cos/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:?????????]
model/ada_cos/ConstConst*
_output_shapes
:*
dtype0*
valueB: v
model/ada_cos/B_avgMeanmodel/ada_cos/Sum:output:0model/ada_cos/Const:output:0*
T0*
_output_shapes
: `
model/ada_cos/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
value	B :?
model/ada_cos/ArgMaxArgMaxonehot_label'model/ada_cos/ArgMax/dimension:output:0*
T0*#
_output_shapes
:?????????`
model/ada_cos/theta_class/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
model/ada_cos/theta_classGatherV2model/ada_cos/Acos:y:0model/ada_cos/ArgMax:output:0'model/ada_cos/theta_class/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*(
_output_shapes
:??????????a
model/ada_cos/percentile/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2
model/ada_cos/percentile/CastCast(model/ada_cos/percentile/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: y
&model/ada_cos/percentile/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
 model/ada_cos/percentile/ReshapeReshape"model/ada_cos/theta_class:output:0/model/ada_cos/percentile/Reshape/shape:output:0*
T0*#
_output_shapes
:?????????k
"model/ada_cos/percentile/truediv/yConst*
_output_shapes
: *
dtype0*
valueB 2      Y@?
 model/ada_cos/percentile/truedivRealDiv!model/ada_cos/percentile/Cast:y:0+model/ada_cos/percentile/truediv/y:output:0*
T0*
_output_shapes
: m
"model/ada_cos/percentile/sort/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
!model/ada_cos/percentile/sort/NegNeg)model/ada_cos/percentile/Reshape:output:0*
T0*#
_output_shapes
:?????????x
#model/ada_cos/percentile/sort/ShapeShape%model/ada_cos/percentile/sort/Neg:y:0*
T0*
_output_shapes
:?
1model/ada_cos/percentile/sort/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????}
3model/ada_cos/percentile/sort/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: }
3model/ada_cos/percentile/sort/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+model/ada_cos/percentile/sort/strided_sliceStridedSlice,model/ada_cos/percentile/sort/Shape:output:0:model/ada_cos/percentile/sort/strided_slice/stack:output:0<model/ada_cos/percentile/sort/strided_slice/stack_1:output:0<model/ada_cos/percentile/sort/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"model/ada_cos/percentile/sort/RankConst*
_output_shapes
: *
dtype0*
value	B :?
$model/ada_cos/percentile/sort/TopKV2TopKV2%model/ada_cos/percentile/sort/Neg:y:04model/ada_cos/percentile/sort/strided_slice:output:0*
T0*2
_output_shapes 
:?????????:??????????
#model/ada_cos/percentile/sort/Neg_1Neg-model/ada_cos/percentile/sort/TopKV2:values:0*
T0*#
_output_shapes
:?????????w
model/ada_cos/percentile/ShapeShape)model/ada_cos/percentile/Reshape:output:0*
T0*
_output_shapes
:
,model/ada_cos/percentile/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????x
.model/ada_cos/percentile/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: x
.model/ada_cos/percentile/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&model/ada_cos/percentile/strided_sliceStridedSlice'model/ada_cos/percentile/Shape:output:05model/ada_cos/percentile/strided_slice/stack:output:07model/ada_cos/percentile/strided_slice/stack_1:output:07model/ada_cos/percentile/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
model/ada_cos/percentile/Cast_1Cast/model/ada_cos/percentile/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: g
model/ada_cos/percentile/sub/yConst*
_output_shapes
: *
dtype0*
valueB 2      ???
model/ada_cos/percentile/subSub#model/ada_cos/percentile/Cast_1:y:0'model/ada_cos/percentile/sub/y:output:0*
T0*
_output_shapes
: ?
model/ada_cos/percentile/mulMul model/ada_cos/percentile/sub:z:0$model/ada_cos/percentile/truediv:z:0*
T0*
_output_shapes
: j
model/ada_cos/percentile/RoundRound model/ada_cos/percentile/mul:z:0*
T0*
_output_shapes
: {
model/ada_cos/percentile/Cast_2Cast"model/ada_cos/percentile/Round:y:0*

DstT0*

SrcT0*
_output_shapes
: y
 model/ada_cos/percentile/Shape_1Shape)model/ada_cos/percentile/Reshape:output:0*
T0*
_output_shapes
:?
.model/ada_cos/percentile/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????z
0model/ada_cos/percentile/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: z
0model/ada_cos/percentile/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
(model/ada_cos/percentile/strided_slice_1StridedSlice)model/ada_cos/percentile/Shape_1:output:07model/ada_cos/percentile/strided_slice_1/stack:output:09model/ada_cos/percentile/strided_slice_1/stack_1:output:09model/ada_cos/percentile/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 model/ada_cos/percentile/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
model/ada_cos/percentile/sub_1Sub1model/ada_cos/percentile/strided_slice_1:output:0)model/ada_cos/percentile/sub_1/y:output:0*
T0*
_output_shapes
: ?
.model/ada_cos/percentile/clip_by_value/MinimumMinimum#model/ada_cos/percentile/Cast_2:y:0"model/ada_cos/percentile/sub_1:z:0*
T0*
_output_shapes
: j
(model/ada_cos/percentile/clip_by_value/yConst*
_output_shapes
: *
dtype0*
value	B : ?
&model/ada_cos/percentile/clip_by_valueMaximum2model/ada_cos/percentile/clip_by_value/Minimum:z:01model/ada_cos/percentile/clip_by_value/y:output:0*
T0*
_output_shapes
: q
&model/ada_cos/percentile/GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
!model/ada_cos/percentile/GatherV2GatherV2'model/ada_cos/percentile/sort/Neg_1:y:0*model/ada_cos/percentile/clip_by_value:z:0/model/ada_cos/percentile/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: ~
model/ada_cos/percentile/IsNanIsNan"model/ada_cos/theta_class:output:0*
T0*(
_output_shapes
:??????????o
model/ada_cos/percentile/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
model/ada_cos/percentile/AnyAny"model/ada_cos/percentile/IsNan:y:0'model/ada_cos/percentile/Const:output:0*
_output_shapes
: k
(model/ada_cos/percentile/Reshape_1/shapeConst*
_output_shapes
: *
dtype0*
valueB ?
"model/ada_cos/percentile/Reshape_1Reshape%model/ada_cos/percentile/Any:output:01model/ada_cos/percentile/Reshape_1/shape:output:0*
T0
*
_output_shapes
: h
#model/ada_cos/percentile/SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *  ??
!model/ada_cos/percentile/SelectV2SelectV2+model/ada_cos/percentile/Reshape_1:output:0,model/ada_cos/percentile/SelectV2/t:output:0*model/ada_cos/percentile/GatherV2:output:0*
T0*
_output_shapes
: q
/model/ada_cos/percentile/rotate_transpose/shiftConst*
_output_shapes
: *
dtype0*
value	B : x
Zmodel/ada_cos/percentile/rotate_transpose/assert_integer/statically_determined_was_integerNoOp*
_output_shapes
 {
model/ada_cos/LogLogmodel/ada_cos/B_avg:output:0"^model/ada_cos/percentile/SelectV2*
T0*
_output_shapes
: ?
model/ada_cos/Minimum/xConst^model/ada_cos/B_avg"^model/ada_cos/percentile/SelectV2*
_output_shapes
: *
dtype0*
valueB
 *?I??
model/ada_cos/MinimumMinimum model/ada_cos/Minimum/x:output:0*model/ada_cos/percentile/SelectV2:output:0*
T0*
_output_shapes
: T
model/ada_cos/CosCosmodel/ada_cos/Minimum:z:0*
T0*
_output_shapes
: o
model/ada_cos/truedivRealDivmodel/ada_cos/Log:y:0model/ada_cos/Cos:y:0*
T0*
_output_shapes
: ?
model/ada_cos/AssignVariableOpAssignVariableOp%model_ada_cos_readvariableop_resourcemodel/ada_cos/truediv:z:0^model/ada_cos/ReadVariableOp*
_output_shapes
 *
dtype0?
model/ada_cos/ReadVariableOp_1ReadVariableOp%model_ada_cos_readvariableop_resource^model/ada_cos/AssignVariableOp^model/ada_cos/B_avg"^model/ada_cos/percentile/SelectV2*
_output_shapes
: *
dtype0?
model/ada_cos/mul_1Mul&model/ada_cos/ReadVariableOp_1:value:0model/ada_cos/matmul:product:0*
T0*(
_output_shapes
:??????????l
model/ada_cos/SoftmaxSoftmaxmodel/ada_cos/mul_1:z:0*
T0*(
_output_shapes
:??????????o
IdentityIdentitymodel/ada_cos/Softmax:softmax:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^model/ada_cos/AssignVariableOp^model/ada_cos/ReadVariableOp^model/ada_cos/ReadVariableOp_1,^model/ada_cos/l2_normalize_1/ReadVariableOp3^model/batch_normalization/batchnorm/ReadVariableOp5^model/batch_normalization/batchnorm/ReadVariableOp_15^model/batch_normalization/batchnorm/ReadVariableOp_27^model/batch_normalization/batchnorm/mul/ReadVariableOp5^model/batch_normalization_1/batchnorm/ReadVariableOp7^model/batch_normalization_1/batchnorm/ReadVariableOp_17^model/batch_normalization_1/batchnorm/ReadVariableOp_29^model/batch_normalization_1/batchnorm/mul/ReadVariableOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:??????????:??????????: : : : : : : : : : : : : : 2@
model/ada_cos/AssignVariableOpmodel/ada_cos/AssignVariableOp2<
model/ada_cos/ReadVariableOpmodel/ada_cos/ReadVariableOp2@
model/ada_cos/ReadVariableOp_1model/ada_cos/ReadVariableOp_12Z
+model/ada_cos/l2_normalize_1/ReadVariableOp+model/ada_cos/l2_normalize_1/ReadVariableOp2h
2model/batch_normalization/batchnorm/ReadVariableOp2model/batch_normalization/batchnorm/ReadVariableOp2l
4model/batch_normalization/batchnorm/ReadVariableOp_14model/batch_normalization/batchnorm/ReadVariableOp_12l
4model/batch_normalization/batchnorm/ReadVariableOp_24model/batch_normalization/batchnorm/ReadVariableOp_22p
6model/batch_normalization/batchnorm/mul/ReadVariableOp6model/batch_normalization/batchnorm/mul/ReadVariableOp2l
4model/batch_normalization_1/batchnorm/ReadVariableOp4model/batch_normalization_1/batchnorm/ReadVariableOp2p
6model/batch_normalization_1/batchnorm/ReadVariableOp_16model/batch_normalization_1/batchnorm/ReadVariableOp_12p
6model/batch_normalization_1/batchnorm/ReadVariableOp_26model/batch_normalization_1/batchnorm/ReadVariableOp_22t
8model/batch_normalization_1/batchnorm/mul/ReadVariableOp8model/batch_normalization_1/batchnorm/mul/ReadVariableOp2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2J
#model/dense_1/MatMul/ReadVariableOp#model/dense_1/MatMul/ReadVariableOp:U Q
(
_output_shapes
:??????????
%
_user_specified_namedense_input:VR
(
_output_shapes
:??????????
&
_user_specified_nameonehot_label
?+
?
A__inference_model_layer_call_and_return_conditional_losses_284547
dense_input
onehot_label 
dense_284506:
??
dense_284508:	?)
batch_normalization_284511:	?)
batch_normalization_284513:	?)
batch_normalization_284515:	?)
batch_normalization_284517:	?"
dense_1_284521:
??
dense_1_284523:	?+
batch_normalization_1_284526:	?+
batch_normalization_1_284528:	?+
batch_normalization_1_284530:	?+
batch_normalization_1_284532:	?"
ada_cos_284535:
??
ada_cos_284537: 
identity??ada_cos/StatefulPartitionedCall?+ada_cos/W/Regularizer/Square/ReadVariableOp?+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCalldense_inputdense_284506dense_284508*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_284104?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_284511batch_normalization_284513batch_normalization_284515batch_normalization_284517*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_283944?
dropout/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_284124?
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_284521dense_1_284523*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_284137?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_1_284526batch_normalization_1_284528batch_normalization_1_284530batch_normalization_1_284532*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_284026?
ada_cos/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0onehot_labelada_cos_284535ada_cos_284537*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_ada_cos_layer_call_and_return_conditional_losses_284257|
+ada_cos/W/Regularizer/Square/ReadVariableOpReadVariableOpada_cos_284535* 
_output_shapes
:
??*
dtype0?
ada_cos/W/Regularizer/SquareSquare3ada_cos/W/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??l
ada_cos/W/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
ada_cos/W/Regularizer/SumSum ada_cos/W/Regularizer/Square:y:0$ada_cos/W/Regularizer/Const:output:0*
T0*
_output_shapes
: `
ada_cos/W/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
ada_cos/W/Regularizer/mulMul$ada_cos/W/Regularizer/mul/x:output:0"ada_cos/W/Regularizer/Sum:output:0*
T0*
_output_shapes
: x
IdentityIdentity(ada_cos/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp ^ada_cos/StatefulPartitionedCall,^ada_cos/W/Regularizer/Square/ReadVariableOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:??????????:??????????: : : : : : : : : : : : : : 2B
ada_cos/StatefulPartitionedCallada_cos/StatefulPartitionedCall2Z
+ada_cos/W/Regularizer/Square/ReadVariableOp+ada_cos/W/Regularizer/Square/ReadVariableOp2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:U Q
(
_output_shapes
:??????????
%
_user_specified_namedense_input:VR
(
_output_shapes
:??????????
&
_user_specified_nameonehot_label
?%
?
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_284073

inputs6
'assignmovingavg_readvariableop_resource:	?8
)assignmovingavg_1_readvariableop_resource:	?4
%batchnorm_mul_readvariableop_resource:	?0
!batchnorm_readvariableop_resource:	?
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	??
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:??????????l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:?y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:??
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:??
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
A__inference_model_layer_call_and_return_conditional_losses_285011
inputs_0
inputs_18
$dense_matmul_readvariableop_resource:
??4
%dense_biasadd_readvariableop_resource:	?J
;batch_normalization_assignmovingavg_readvariableop_resource:	?L
=batch_normalization_assignmovingavg_1_readvariableop_resource:	?H
9batch_normalization_batchnorm_mul_readvariableop_resource:	?D
5batch_normalization_batchnorm_readvariableop_resource:	?:
&dense_1_matmul_readvariableop_resource:
??6
'dense_1_biasadd_readvariableop_resource:	?L
=batch_normalization_1_assignmovingavg_readvariableop_resource:	?N
?batch_normalization_1_assignmovingavg_1_readvariableop_resource:	?J
;batch_normalization_1_batchnorm_mul_readvariableop_resource:	?F
7batch_normalization_1_batchnorm_readvariableop_resource:	?B
.ada_cos_l2_normalize_1_readvariableop_resource:
??)
ada_cos_readvariableop_resource: 
identity??ada_cos/AssignVariableOp?ada_cos/ReadVariableOp?ada_cos/ReadVariableOp_1?+ada_cos/W/Regularizer/Square/ReadVariableOp?%ada_cos/l2_normalize_1/ReadVariableOp?#batch_normalization/AssignMovingAvg?2batch_normalization/AssignMovingAvg/ReadVariableOp?%batch_normalization/AssignMovingAvg_1?4batch_normalization/AssignMovingAvg_1/ReadVariableOp?,batch_normalization/batchnorm/ReadVariableOp?0batch_normalization/batchnorm/mul/ReadVariableOp?%batch_normalization_1/AssignMovingAvg?4batch_normalization_1/AssignMovingAvg/ReadVariableOp?'batch_normalization_1/AssignMovingAvg_1?6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp?.batch_normalization_1/batchnorm/ReadVariableOp?2batch_normalization_1/batchnorm/mul/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0x
dense/MatMulMatMulinputs_0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????|
2batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
 batch_normalization/moments/meanMeandense/Relu:activations:0;batch_normalization/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(?
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:	??
-batch_normalization/moments/SquaredDifferenceSquaredDifferencedense/Relu:activations:01batch_normalization/moments/StopGradient:output:0*
T0*(
_output_shapes
:???????????
6batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
$batch_normalization/moments/varianceMean1batch_normalization/moments/SquaredDifference:z:0?batch_normalization/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(?
#batch_normalization/moments/SqueezeSqueeze)batch_normalization/moments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 ?
%batch_normalization/moments/Squeeze_1Squeeze-batch_normalization/moments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 n
)batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp;batch_normalization_assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization/AssignMovingAvg/subSub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:0,batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes	
:??
'batch_normalization/AssignMovingAvg/mulMul+batch_normalization/AssignMovingAvg/sub:z:02batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:??
#batch_normalization/AssignMovingAvgAssignSubVariableOp;batch_normalization_assignmovingavg_readvariableop_resource+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0p
+batch_normalization/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
)batch_normalization/AssignMovingAvg_1/subSub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0.batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:??
)batch_normalization/AssignMovingAvg_1/mulMul-batch_normalization/AssignMovingAvg_1/sub:z:04batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:??
%batch_normalization/AssignMovingAvg_1AssignSubVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0h
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
!batch_normalization/batchnorm/addAddV2.batch_normalization/moments/Squeeze_1:output:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?y
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:??
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:??
#batch_normalization/batchnorm/mul_1Muldense/Relu:activations:0%batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:???????????
#batch_normalization/batchnorm/mul_2Mul,batch_normalization/moments/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:??
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!batch_normalization/batchnorm/subSub4batch_normalization/batchnorm/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:??
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?????
dropout/dropout/MulMul'batch_normalization/batchnorm/add_1:z:0dropout/dropout/Const:output:0*
T0*(
_output_shapes
:??????????l
dropout/dropout/ShapeShape'batch_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:???????????
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_1/MatMulMatMuldropout/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????a
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????~
4batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
"batch_normalization_1/moments/meanMeandense_1/Relu:activations:0=batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(?
*batch_normalization_1/moments/StopGradientStopGradient+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes
:	??
/batch_normalization_1/moments/SquaredDifferenceSquaredDifferencedense_1/Relu:activations:03batch_normalization_1/moments/StopGradient:output:0*
T0*(
_output_shapes
:???????????
8batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
&batch_normalization_1/moments/varianceMean3batch_normalization_1/moments/SquaredDifference:z:0Abatch_normalization_1/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(?
%batch_normalization_1/moments/SqueezeSqueeze+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 ?
'batch_normalization_1/moments/Squeeze_1Squeeze/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 p
+batch_normalization_1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype0?
)batch_normalization_1/AssignMovingAvg/subSub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_1/moments/Squeeze:output:0*
T0*
_output_shapes	
:??
)batch_normalization_1/AssignMovingAvg/mulMul-batch_normalization_1/AssignMovingAvg/sub:z:04batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:??
%batch_normalization_1/AssignMovingAvgAssignSubVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource-batch_normalization_1/AssignMovingAvg/mul:z:05^batch_normalization_1/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0r
-batch_normalization_1/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
+batch_normalization_1/AssignMovingAvg_1/subSub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_1/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:??
+batch_normalization_1/AssignMovingAvg_1/mulMul/batch_normalization_1/AssignMovingAvg_1/sub:z:06batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:??
'batch_normalization_1/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource/batch_normalization_1/AssignMovingAvg_1/mul:z:07^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0j
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
#batch_normalization_1/batchnorm/addAddV20batch_normalization_1/moments/Squeeze_1:output:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?}
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:??
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype0?
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:??
%batch_normalization_1/batchnorm/mul_1Muldense_1/Relu:activations:0'batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:???????????
%batch_normalization_1/batchnorm/mul_2Mul.batch_normalization_1/moments/Squeeze:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:??
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype0?
#batch_normalization_1/batchnorm/subSub6batch_normalization_1/batchnorm/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:??
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:???????????
ada_cos/l2_normalize/SquareSquare)batch_normalization_1/batchnorm/add_1:z:0*
T0*(
_output_shapes
:??????????l
*ada_cos/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
ada_cos/l2_normalize/SumSumada_cos/l2_normalize/Square:y:03ada_cos/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(c
ada_cos/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼?+?
ada_cos/l2_normalize/MaximumMaximum!ada_cos/l2_normalize/Sum:output:0'ada_cos/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:?????????w
ada_cos/l2_normalize/RsqrtRsqrt ada_cos/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:??????????
ada_cos/l2_normalizeMul)batch_normalization_1/batchnorm/add_1:z:0ada_cos/l2_normalize/Rsqrt:y:0*
T0*(
_output_shapes
:???????????
%ada_cos/l2_normalize_1/ReadVariableOpReadVariableOp.ada_cos_l2_normalize_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
ada_cos/l2_normalize_1/SquareSquare-ada_cos/l2_normalize_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??n
,ada_cos/l2_normalize_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : ?
ada_cos/l2_normalize_1/SumSum!ada_cos/l2_normalize_1/Square:y:05ada_cos/l2_normalize_1/Sum/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(e
 ada_cos/l2_normalize_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼?+?
ada_cos/l2_normalize_1/MaximumMaximum#ada_cos/l2_normalize_1/Sum:output:0)ada_cos/l2_normalize_1/Maximum/y:output:0*
T0*
_output_shapes
:	?s
ada_cos/l2_normalize_1/RsqrtRsqrt"ada_cos/l2_normalize_1/Maximum:z:0*
T0*
_output_shapes
:	??
ada_cos/l2_normalize_1Mul-ada_cos/l2_normalize_1/ReadVariableOp:value:0 ada_cos/l2_normalize_1/Rsqrt:y:0*
T0* 
_output_shapes
:
???
ada_cos/matmulMatMulada_cos/l2_normalize:z:0ada_cos/l2_normalize_1:z:0*
T0*(
_output_shapes
:??????????d
ada_cos/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *????
ada_cos/clip_by_value/MinimumMinimumada_cos/matmul:product:0(ada_cos/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????\
ada_cos/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *????
ada_cos/clip_by_valueMaximum!ada_cos/clip_by_value/Minimum:z:0 ada_cos/clip_by_value/y:output:0*
T0*(
_output_shapes
:??????????b
ada_cos/AcosAcosada_cos/clip_by_value:z:0*
T0*(
_output_shapes
:??????????P
ada_cos/Less/yConst*
_output_shapes
: *
dtype0*
value	B :j
ada_cos/LessLessinputs_1ada_cos/Less/y:output:0*
T0*(
_output_shapes
:??????????n
ada_cos/ReadVariableOpReadVariableOpada_cos_readvariableop_resource*
_output_shapes
: *
dtype0
ada_cos/mulMulada_cos/ReadVariableOp:value:0ada_cos/matmul:product:0*
T0*(
_output_shapes
:??????????V
ada_cos/ExpExpada_cos/mul:z:0*
T0*(
_output_shapes
:??????????l
ada_cos/zeros_like	ZerosLikeada_cos/matmul:product:0*
T0*(
_output_shapes
:???????????
ada_cos/SelectV2SelectV2ada_cos/Less:z:0ada_cos/Exp:y:0ada_cos/zeros_like:y:0*
T0*(
_output_shapes
:??????????_
ada_cos/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
ada_cos/SumSumada_cos/SelectV2:output:0&ada_cos/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:?????????W
ada_cos/ConstConst*
_output_shapes
:*
dtype0*
valueB: d
ada_cos/B_avgMeanada_cos/Sum:output:0ada_cos/Const:output:0*
T0*
_output_shapes
: Z
ada_cos/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
value	B :s
ada_cos/ArgMaxArgMaxinputs_1!ada_cos/ArgMax/dimension:output:0*
T0*#
_output_shapes
:?????????Z
ada_cos/theta_class/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
ada_cos/theta_classGatherV2ada_cos/Acos:y:0ada_cos/ArgMax:output:0!ada_cos/theta_class/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*(
_output_shapes
:??????????[
ada_cos/percentile/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2s
ada_cos/percentile/CastCast"ada_cos/percentile/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: s
 ada_cos/percentile/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
ada_cos/percentile/ReshapeReshapeada_cos/theta_class:output:0)ada_cos/percentile/Reshape/shape:output:0*
T0*#
_output_shapes
:?????????e
ada_cos/percentile/truediv/yConst*
_output_shapes
: *
dtype0*
valueB 2      Y@?
ada_cos/percentile/truedivRealDivada_cos/percentile/Cast:y:0%ada_cos/percentile/truediv/y:output:0*
T0*
_output_shapes
: g
ada_cos/percentile/sort/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????u
ada_cos/percentile/sort/NegNeg#ada_cos/percentile/Reshape:output:0*
T0*#
_output_shapes
:?????????l
ada_cos/percentile/sort/ShapeShapeada_cos/percentile/sort/Neg:y:0*
T0*
_output_shapes
:~
+ada_cos/percentile/sort/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????w
-ada_cos/percentile/sort/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-ada_cos/percentile/sort/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%ada_cos/percentile/sort/strided_sliceStridedSlice&ada_cos/percentile/sort/Shape:output:04ada_cos/percentile/sort/strided_slice/stack:output:06ada_cos/percentile/sort/strided_slice/stack_1:output:06ada_cos/percentile/sort/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
ada_cos/percentile/sort/RankConst*
_output_shapes
: *
dtype0*
value	B :?
ada_cos/percentile/sort/TopKV2TopKV2ada_cos/percentile/sort/Neg:y:0.ada_cos/percentile/sort/strided_slice:output:0*
T0*2
_output_shapes 
:?????????:?????????{
ada_cos/percentile/sort/Neg_1Neg'ada_cos/percentile/sort/TopKV2:values:0*
T0*#
_output_shapes
:?????????k
ada_cos/percentile/ShapeShape#ada_cos/percentile/Reshape:output:0*
T0*
_output_shapes
:y
&ada_cos/percentile/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????r
(ada_cos/percentile/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: r
(ada_cos/percentile/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 ada_cos/percentile/strided_sliceStridedSlice!ada_cos/percentile/Shape:output:0/ada_cos/percentile/strided_slice/stack:output:01ada_cos/percentile/strided_slice/stack_1:output:01ada_cos/percentile/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
ada_cos/percentile/Cast_1Cast)ada_cos/percentile/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: a
ada_cos/percentile/sub/yConst*
_output_shapes
: *
dtype0*
valueB 2      ???
ada_cos/percentile/subSubada_cos/percentile/Cast_1:y:0!ada_cos/percentile/sub/y:output:0*
T0*
_output_shapes
: z
ada_cos/percentile/mulMulada_cos/percentile/sub:z:0ada_cos/percentile/truediv:z:0*
T0*
_output_shapes
: ^
ada_cos/percentile/RoundRoundada_cos/percentile/mul:z:0*
T0*
_output_shapes
: o
ada_cos/percentile/Cast_2Castada_cos/percentile/Round:y:0*

DstT0*

SrcT0*
_output_shapes
: m
ada_cos/percentile/Shape_1Shape#ada_cos/percentile/Reshape:output:0*
T0*
_output_shapes
:{
(ada_cos/percentile/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????t
*ada_cos/percentile/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: t
*ada_cos/percentile/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"ada_cos/percentile/strided_slice_1StridedSlice#ada_cos/percentile/Shape_1:output:01ada_cos/percentile/strided_slice_1/stack:output:03ada_cos/percentile/strided_slice_1/stack_1:output:03ada_cos/percentile/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
ada_cos/percentile/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
ada_cos/percentile/sub_1Sub+ada_cos/percentile/strided_slice_1:output:0#ada_cos/percentile/sub_1/y:output:0*
T0*
_output_shapes
: ?
(ada_cos/percentile/clip_by_value/MinimumMinimumada_cos/percentile/Cast_2:y:0ada_cos/percentile/sub_1:z:0*
T0*
_output_shapes
: d
"ada_cos/percentile/clip_by_value/yConst*
_output_shapes
: *
dtype0*
value	B : ?
 ada_cos/percentile/clip_by_valueMaximum,ada_cos/percentile/clip_by_value/Minimum:z:0+ada_cos/percentile/clip_by_value/y:output:0*
T0*
_output_shapes
: k
 ada_cos/percentile/GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
ada_cos/percentile/GatherV2GatherV2!ada_cos/percentile/sort/Neg_1:y:0$ada_cos/percentile/clip_by_value:z:0)ada_cos/percentile/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: r
ada_cos/percentile/IsNanIsNanada_cos/theta_class:output:0*
T0*(
_output_shapes
:??????????i
ada_cos/percentile/ConstConst*
_output_shapes
:*
dtype0*
valueB"       v
ada_cos/percentile/AnyAnyada_cos/percentile/IsNan:y:0!ada_cos/percentile/Const:output:0*
_output_shapes
: e
"ada_cos/percentile/Reshape_1/shapeConst*
_output_shapes
: *
dtype0*
valueB ?
ada_cos/percentile/Reshape_1Reshapeada_cos/percentile/Any:output:0+ada_cos/percentile/Reshape_1/shape:output:0*
T0
*
_output_shapes
: b
ada_cos/percentile/SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *  ??
ada_cos/percentile/SelectV2SelectV2%ada_cos/percentile/Reshape_1:output:0&ada_cos/percentile/SelectV2/t:output:0$ada_cos/percentile/GatherV2:output:0*
T0*
_output_shapes
: k
)ada_cos/percentile/rotate_transpose/shiftConst*
_output_shapes
: *
dtype0*
value	B : r
Tada_cos/percentile/rotate_transpose/assert_integer/statically_determined_was_integerNoOp*
_output_shapes
 i
ada_cos/LogLogada_cos/B_avg:output:0^ada_cos/percentile/SelectV2*
T0*
_output_shapes
: ?
ada_cos/Minimum/xConst^ada_cos/B_avg^ada_cos/percentile/SelectV2*
_output_shapes
: *
dtype0*
valueB
 *?I?}
ada_cos/MinimumMinimumada_cos/Minimum/x:output:0$ada_cos/percentile/SelectV2:output:0*
T0*
_output_shapes
: H
ada_cos/CosCosada_cos/Minimum:z:0*
T0*
_output_shapes
: ]
ada_cos/truedivRealDivada_cos/Log:y:0ada_cos/Cos:y:0*
T0*
_output_shapes
: ?
ada_cos/AssignVariableOpAssignVariableOpada_cos_readvariableop_resourceada_cos/truediv:z:0^ada_cos/ReadVariableOp*
_output_shapes
 *
dtype0?
ada_cos/ReadVariableOp_1ReadVariableOpada_cos_readvariableop_resource^ada_cos/AssignVariableOp^ada_cos/B_avg^ada_cos/percentile/SelectV2*
_output_shapes
: *
dtype0?
ada_cos/mul_1Mul ada_cos/ReadVariableOp_1:value:0ada_cos/matmul:product:0*
T0*(
_output_shapes
:??????????`
ada_cos/SoftmaxSoftmaxada_cos/mul_1:z:0*
T0*(
_output_shapes
:???????????
+ada_cos/W/Regularizer/Square/ReadVariableOpReadVariableOp.ada_cos_l2_normalize_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
ada_cos/W/Regularizer/SquareSquare3ada_cos/W/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??l
ada_cos/W/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
ada_cos/W/Regularizer/SumSum ada_cos/W/Regularizer/Square:y:0$ada_cos/W/Regularizer/Const:output:0*
T0*
_output_shapes
: `
ada_cos/W/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
ada_cos/W/Regularizer/mulMul$ada_cos/W/Regularizer/mul/x:output:0"ada_cos/W/Regularizer/Sum:output:0*
T0*
_output_shapes
: i
IdentityIdentityada_cos/Softmax:softmax:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^ada_cos/AssignVariableOp^ada_cos/ReadVariableOp^ada_cos/ReadVariableOp_1,^ada_cos/W/Regularizer/Square/ReadVariableOp&^ada_cos/l2_normalize_1/ReadVariableOp$^batch_normalization/AssignMovingAvg3^batch_normalization/AssignMovingAvg/ReadVariableOp&^batch_normalization/AssignMovingAvg_15^batch_normalization/AssignMovingAvg_1/ReadVariableOp-^batch_normalization/batchnorm/ReadVariableOp1^batch_normalization/batchnorm/mul/ReadVariableOp&^batch_normalization_1/AssignMovingAvg5^batch_normalization_1/AssignMovingAvg/ReadVariableOp(^batch_normalization_1/AssignMovingAvg_17^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp3^batch_normalization_1/batchnorm/mul/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:??????????:??????????: : : : : : : : : : : : : : 24
ada_cos/AssignVariableOpada_cos/AssignVariableOp20
ada_cos/ReadVariableOpada_cos/ReadVariableOp24
ada_cos/ReadVariableOp_1ada_cos/ReadVariableOp_12Z
+ada_cos/W/Regularizer/Square/ReadVariableOp+ada_cos/W/Regularizer/Square/ReadVariableOp2N
%ada_cos/l2_normalize_1/ReadVariableOp%ada_cos/l2_normalize_1/ReadVariableOp2J
#batch_normalization/AssignMovingAvg#batch_normalization/AssignMovingAvg2h
2batch_normalization/AssignMovingAvg/ReadVariableOp2batch_normalization/AssignMovingAvg/ReadVariableOp2N
%batch_normalization/AssignMovingAvg_1%batch_normalization/AssignMovingAvg_12l
4batch_normalization/AssignMovingAvg_1/ReadVariableOp4batch_normalization/AssignMovingAvg_1/ReadVariableOp2\
,batch_normalization/batchnorm/ReadVariableOp,batch_normalization/batchnorm/ReadVariableOp2d
0batch_normalization/batchnorm/mul/ReadVariableOp0batch_normalization/batchnorm/mul/ReadVariableOp2N
%batch_normalization_1/AssignMovingAvg%batch_normalization_1/AssignMovingAvg2l
4batch_normalization_1/AssignMovingAvg/ReadVariableOp4batch_normalization_1/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_1/AssignMovingAvg_1'batch_normalization_1/AssignMovingAvg_12p
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_1/batchnorm/ReadVariableOp.batch_normalization_1/batchnorm/ReadVariableOp2h
2batch_normalization_1/batchnorm/mul/ReadVariableOp2batch_normalization_1/batchnorm/mul/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:R N
(
_output_shapes
:??????????
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/1
?
?
&__inference_dense_layer_call_fn_285056

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_284104p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_285240

inputs0
!batchnorm_readvariableop_resource:	?4
%batchnorm_mul_readvariableop_resource:	?2
#batchnorm_readvariableop_1_resource:	?2
#batchnorm_readvariableop_2_resource:	?
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
(__inference_dense_1_layer_call_fn_285183

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_284137p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
__inference_loss_fn_0_285406H
4ada_cos_w_regularizer_square_readvariableop_resource:
??
identity??+ada_cos/W/Regularizer/Square/ReadVariableOp?
+ada_cos/W/Regularizer/Square/ReadVariableOpReadVariableOp4ada_cos_w_regularizer_square_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
ada_cos/W/Regularizer/SquareSquare3ada_cos/W/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??l
ada_cos/W/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
ada_cos/W/Regularizer/SumSum ada_cos/W/Regularizer/Square:y:0$ada_cos/W/Regularizer/Const:output:0*
T0*
_output_shapes
: `
ada_cos/W/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
ada_cos/W/Regularizer/mulMul$ada_cos/W/Regularizer/mul/x:output:0"ada_cos/W/Regularizer/Sum:output:0*
T0*
_output_shapes
: [
IdentityIdentityada_cos/W/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: t
NoOpNoOp,^ada_cos/W/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2Z
+ada_cos/W/Regularizer/Square/ReadVariableOp+ada_cos/W/Regularizer/Square/ReadVariableOp
?
D
(__inference_dropout_layer_call_fn_285152

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_284124a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
A__inference_dense_layer_call_and_return_conditional_losses_284104

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
A__inference_model_layer_call_and_return_conditional_losses_284824
inputs_0
inputs_18
$dense_matmul_readvariableop_resource:
??4
%dense_biasadd_readvariableop_resource:	?D
5batch_normalization_batchnorm_readvariableop_resource:	?H
9batch_normalization_batchnorm_mul_readvariableop_resource:	?F
7batch_normalization_batchnorm_readvariableop_1_resource:	?F
7batch_normalization_batchnorm_readvariableop_2_resource:	?:
&dense_1_matmul_readvariableop_resource:
??6
'dense_1_biasadd_readvariableop_resource:	?F
7batch_normalization_1_batchnorm_readvariableop_resource:	?J
;batch_normalization_1_batchnorm_mul_readvariableop_resource:	?H
9batch_normalization_1_batchnorm_readvariableop_1_resource:	?H
9batch_normalization_1_batchnorm_readvariableop_2_resource:	?B
.ada_cos_l2_normalize_1_readvariableop_resource:
??)
ada_cos_readvariableop_resource: 
identity??ada_cos/AssignVariableOp?ada_cos/ReadVariableOp?ada_cos/ReadVariableOp_1?+ada_cos/W/Regularizer/Square/ReadVariableOp?%ada_cos/l2_normalize_1/ReadVariableOp?,batch_normalization/batchnorm/ReadVariableOp?.batch_normalization/batchnorm/ReadVariableOp_1?.batch_normalization/batchnorm/ReadVariableOp_2?0batch_normalization/batchnorm/mul/ReadVariableOp?.batch_normalization_1/batchnorm/ReadVariableOp?0batch_normalization_1/batchnorm/ReadVariableOp_1?0batch_normalization_1/batchnorm/ReadVariableOp_2?2batch_normalization_1/batchnorm/mul/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0x
dense/MatMulMatMulinputs_0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype0h
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
!batch_normalization/batchnorm/addAddV24batch_normalization/batchnorm/ReadVariableOp:value:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?y
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:??
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:??
#batch_normalization/batchnorm/mul_1Muldense/Relu:activations:0%batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:???????????
.batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp7batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
#batch_normalization/batchnorm/mul_2Mul6batch_normalization/batchnorm/ReadVariableOp_1:value:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:??
.batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp7batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype0?
!batch_normalization/batchnorm/subSub6batch_normalization/batchnorm/ReadVariableOp_2:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:??
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????x
dropout/IdentityIdentity'batch_normalization/batchnorm/add_1:z:0*
T0*(
_output_shapes
:???????????
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_1/MatMulMatMuldropout/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????a
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype0j
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
#batch_normalization_1/batchnorm/addAddV26batch_normalization_1/batchnorm/ReadVariableOp:value:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?}
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:??
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype0?
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:??
%batch_normalization_1/batchnorm/mul_1Muldense_1/Relu:activations:0'batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:???????????
0batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
%batch_normalization_1/batchnorm/mul_2Mul8batch_normalization_1/batchnorm/ReadVariableOp_1:value:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:??
0batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype0?
#batch_normalization_1/batchnorm/subSub8batch_normalization_1/batchnorm/ReadVariableOp_2:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:??
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:???????????
ada_cos/l2_normalize/SquareSquare)batch_normalization_1/batchnorm/add_1:z:0*
T0*(
_output_shapes
:??????????l
*ada_cos/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
ada_cos/l2_normalize/SumSumada_cos/l2_normalize/Square:y:03ada_cos/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(c
ada_cos/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼?+?
ada_cos/l2_normalize/MaximumMaximum!ada_cos/l2_normalize/Sum:output:0'ada_cos/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:?????????w
ada_cos/l2_normalize/RsqrtRsqrt ada_cos/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:??????????
ada_cos/l2_normalizeMul)batch_normalization_1/batchnorm/add_1:z:0ada_cos/l2_normalize/Rsqrt:y:0*
T0*(
_output_shapes
:???????????
%ada_cos/l2_normalize_1/ReadVariableOpReadVariableOp.ada_cos_l2_normalize_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
ada_cos/l2_normalize_1/SquareSquare-ada_cos/l2_normalize_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??n
,ada_cos/l2_normalize_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : ?
ada_cos/l2_normalize_1/SumSum!ada_cos/l2_normalize_1/Square:y:05ada_cos/l2_normalize_1/Sum/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(e
 ada_cos/l2_normalize_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼?+?
ada_cos/l2_normalize_1/MaximumMaximum#ada_cos/l2_normalize_1/Sum:output:0)ada_cos/l2_normalize_1/Maximum/y:output:0*
T0*
_output_shapes
:	?s
ada_cos/l2_normalize_1/RsqrtRsqrt"ada_cos/l2_normalize_1/Maximum:z:0*
T0*
_output_shapes
:	??
ada_cos/l2_normalize_1Mul-ada_cos/l2_normalize_1/ReadVariableOp:value:0 ada_cos/l2_normalize_1/Rsqrt:y:0*
T0* 
_output_shapes
:
???
ada_cos/matmulMatMulada_cos/l2_normalize:z:0ada_cos/l2_normalize_1:z:0*
T0*(
_output_shapes
:??????????d
ada_cos/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *????
ada_cos/clip_by_value/MinimumMinimumada_cos/matmul:product:0(ada_cos/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????\
ada_cos/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *????
ada_cos/clip_by_valueMaximum!ada_cos/clip_by_value/Minimum:z:0 ada_cos/clip_by_value/y:output:0*
T0*(
_output_shapes
:??????????b
ada_cos/AcosAcosada_cos/clip_by_value:z:0*
T0*(
_output_shapes
:??????????P
ada_cos/Less/yConst*
_output_shapes
: *
dtype0*
value	B :j
ada_cos/LessLessinputs_1ada_cos/Less/y:output:0*
T0*(
_output_shapes
:??????????n
ada_cos/ReadVariableOpReadVariableOpada_cos_readvariableop_resource*
_output_shapes
: *
dtype0
ada_cos/mulMulada_cos/ReadVariableOp:value:0ada_cos/matmul:product:0*
T0*(
_output_shapes
:??????????V
ada_cos/ExpExpada_cos/mul:z:0*
T0*(
_output_shapes
:??????????l
ada_cos/zeros_like	ZerosLikeada_cos/matmul:product:0*
T0*(
_output_shapes
:???????????
ada_cos/SelectV2SelectV2ada_cos/Less:z:0ada_cos/Exp:y:0ada_cos/zeros_like:y:0*
T0*(
_output_shapes
:??????????_
ada_cos/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
ada_cos/SumSumada_cos/SelectV2:output:0&ada_cos/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:?????????W
ada_cos/ConstConst*
_output_shapes
:*
dtype0*
valueB: d
ada_cos/B_avgMeanada_cos/Sum:output:0ada_cos/Const:output:0*
T0*
_output_shapes
: Z
ada_cos/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
value	B :s
ada_cos/ArgMaxArgMaxinputs_1!ada_cos/ArgMax/dimension:output:0*
T0*#
_output_shapes
:?????????Z
ada_cos/theta_class/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
ada_cos/theta_classGatherV2ada_cos/Acos:y:0ada_cos/ArgMax:output:0!ada_cos/theta_class/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*(
_output_shapes
:??????????[
ada_cos/percentile/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2s
ada_cos/percentile/CastCast"ada_cos/percentile/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: s
 ada_cos/percentile/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
ada_cos/percentile/ReshapeReshapeada_cos/theta_class:output:0)ada_cos/percentile/Reshape/shape:output:0*
T0*#
_output_shapes
:?????????e
ada_cos/percentile/truediv/yConst*
_output_shapes
: *
dtype0*
valueB 2      Y@?
ada_cos/percentile/truedivRealDivada_cos/percentile/Cast:y:0%ada_cos/percentile/truediv/y:output:0*
T0*
_output_shapes
: g
ada_cos/percentile/sort/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????u
ada_cos/percentile/sort/NegNeg#ada_cos/percentile/Reshape:output:0*
T0*#
_output_shapes
:?????????l
ada_cos/percentile/sort/ShapeShapeada_cos/percentile/sort/Neg:y:0*
T0*
_output_shapes
:~
+ada_cos/percentile/sort/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????w
-ada_cos/percentile/sort/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-ada_cos/percentile/sort/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%ada_cos/percentile/sort/strided_sliceStridedSlice&ada_cos/percentile/sort/Shape:output:04ada_cos/percentile/sort/strided_slice/stack:output:06ada_cos/percentile/sort/strided_slice/stack_1:output:06ada_cos/percentile/sort/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
ada_cos/percentile/sort/RankConst*
_output_shapes
: *
dtype0*
value	B :?
ada_cos/percentile/sort/TopKV2TopKV2ada_cos/percentile/sort/Neg:y:0.ada_cos/percentile/sort/strided_slice:output:0*
T0*2
_output_shapes 
:?????????:?????????{
ada_cos/percentile/sort/Neg_1Neg'ada_cos/percentile/sort/TopKV2:values:0*
T0*#
_output_shapes
:?????????k
ada_cos/percentile/ShapeShape#ada_cos/percentile/Reshape:output:0*
T0*
_output_shapes
:y
&ada_cos/percentile/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????r
(ada_cos/percentile/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: r
(ada_cos/percentile/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 ada_cos/percentile/strided_sliceStridedSlice!ada_cos/percentile/Shape:output:0/ada_cos/percentile/strided_slice/stack:output:01ada_cos/percentile/strided_slice/stack_1:output:01ada_cos/percentile/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
ada_cos/percentile/Cast_1Cast)ada_cos/percentile/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: a
ada_cos/percentile/sub/yConst*
_output_shapes
: *
dtype0*
valueB 2      ???
ada_cos/percentile/subSubada_cos/percentile/Cast_1:y:0!ada_cos/percentile/sub/y:output:0*
T0*
_output_shapes
: z
ada_cos/percentile/mulMulada_cos/percentile/sub:z:0ada_cos/percentile/truediv:z:0*
T0*
_output_shapes
: ^
ada_cos/percentile/RoundRoundada_cos/percentile/mul:z:0*
T0*
_output_shapes
: o
ada_cos/percentile/Cast_2Castada_cos/percentile/Round:y:0*

DstT0*

SrcT0*
_output_shapes
: m
ada_cos/percentile/Shape_1Shape#ada_cos/percentile/Reshape:output:0*
T0*
_output_shapes
:{
(ada_cos/percentile/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????t
*ada_cos/percentile/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: t
*ada_cos/percentile/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"ada_cos/percentile/strided_slice_1StridedSlice#ada_cos/percentile/Shape_1:output:01ada_cos/percentile/strided_slice_1/stack:output:03ada_cos/percentile/strided_slice_1/stack_1:output:03ada_cos/percentile/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
ada_cos/percentile/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
ada_cos/percentile/sub_1Sub+ada_cos/percentile/strided_slice_1:output:0#ada_cos/percentile/sub_1/y:output:0*
T0*
_output_shapes
: ?
(ada_cos/percentile/clip_by_value/MinimumMinimumada_cos/percentile/Cast_2:y:0ada_cos/percentile/sub_1:z:0*
T0*
_output_shapes
: d
"ada_cos/percentile/clip_by_value/yConst*
_output_shapes
: *
dtype0*
value	B : ?
 ada_cos/percentile/clip_by_valueMaximum,ada_cos/percentile/clip_by_value/Minimum:z:0+ada_cos/percentile/clip_by_value/y:output:0*
T0*
_output_shapes
: k
 ada_cos/percentile/GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
ada_cos/percentile/GatherV2GatherV2!ada_cos/percentile/sort/Neg_1:y:0$ada_cos/percentile/clip_by_value:z:0)ada_cos/percentile/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: r
ada_cos/percentile/IsNanIsNanada_cos/theta_class:output:0*
T0*(
_output_shapes
:??????????i
ada_cos/percentile/ConstConst*
_output_shapes
:*
dtype0*
valueB"       v
ada_cos/percentile/AnyAnyada_cos/percentile/IsNan:y:0!ada_cos/percentile/Const:output:0*
_output_shapes
: e
"ada_cos/percentile/Reshape_1/shapeConst*
_output_shapes
: *
dtype0*
valueB ?
ada_cos/percentile/Reshape_1Reshapeada_cos/percentile/Any:output:0+ada_cos/percentile/Reshape_1/shape:output:0*
T0
*
_output_shapes
: b
ada_cos/percentile/SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *  ??
ada_cos/percentile/SelectV2SelectV2%ada_cos/percentile/Reshape_1:output:0&ada_cos/percentile/SelectV2/t:output:0$ada_cos/percentile/GatherV2:output:0*
T0*
_output_shapes
: k
)ada_cos/percentile/rotate_transpose/shiftConst*
_output_shapes
: *
dtype0*
value	B : r
Tada_cos/percentile/rotate_transpose/assert_integer/statically_determined_was_integerNoOp*
_output_shapes
 i
ada_cos/LogLogada_cos/B_avg:output:0^ada_cos/percentile/SelectV2*
T0*
_output_shapes
: ?
ada_cos/Minimum/xConst^ada_cos/B_avg^ada_cos/percentile/SelectV2*
_output_shapes
: *
dtype0*
valueB
 *?I?}
ada_cos/MinimumMinimumada_cos/Minimum/x:output:0$ada_cos/percentile/SelectV2:output:0*
T0*
_output_shapes
: H
ada_cos/CosCosada_cos/Minimum:z:0*
T0*
_output_shapes
: ]
ada_cos/truedivRealDivada_cos/Log:y:0ada_cos/Cos:y:0*
T0*
_output_shapes
: ?
ada_cos/AssignVariableOpAssignVariableOpada_cos_readvariableop_resourceada_cos/truediv:z:0^ada_cos/ReadVariableOp*
_output_shapes
 *
dtype0?
ada_cos/ReadVariableOp_1ReadVariableOpada_cos_readvariableop_resource^ada_cos/AssignVariableOp^ada_cos/B_avg^ada_cos/percentile/SelectV2*
_output_shapes
: *
dtype0?
ada_cos/mul_1Mul ada_cos/ReadVariableOp_1:value:0ada_cos/matmul:product:0*
T0*(
_output_shapes
:??????????`
ada_cos/SoftmaxSoftmaxada_cos/mul_1:z:0*
T0*(
_output_shapes
:???????????
+ada_cos/W/Regularizer/Square/ReadVariableOpReadVariableOp.ada_cos_l2_normalize_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
ada_cos/W/Regularizer/SquareSquare3ada_cos/W/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??l
ada_cos/W/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
ada_cos/W/Regularizer/SumSum ada_cos/W/Regularizer/Square:y:0$ada_cos/W/Regularizer/Const:output:0*
T0*
_output_shapes
: `
ada_cos/W/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
ada_cos/W/Regularizer/mulMul$ada_cos/W/Regularizer/mul/x:output:0"ada_cos/W/Regularizer/Sum:output:0*
T0*
_output_shapes
: i
IdentityIdentityada_cos/Softmax:softmax:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^ada_cos/AssignVariableOp^ada_cos/ReadVariableOp^ada_cos/ReadVariableOp_1,^ada_cos/W/Regularizer/Square/ReadVariableOp&^ada_cos/l2_normalize_1/ReadVariableOp-^batch_normalization/batchnorm/ReadVariableOp/^batch_normalization/batchnorm/ReadVariableOp_1/^batch_normalization/batchnorm/ReadVariableOp_21^batch_normalization/batchnorm/mul/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp1^batch_normalization_1/batchnorm/ReadVariableOp_11^batch_normalization_1/batchnorm/ReadVariableOp_23^batch_normalization_1/batchnorm/mul/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:??????????:??????????: : : : : : : : : : : : : : 24
ada_cos/AssignVariableOpada_cos/AssignVariableOp20
ada_cos/ReadVariableOpada_cos/ReadVariableOp24
ada_cos/ReadVariableOp_1ada_cos/ReadVariableOp_12Z
+ada_cos/W/Regularizer/Square/ReadVariableOp+ada_cos/W/Regularizer/Square/ReadVariableOp2N
%ada_cos/l2_normalize_1/ReadVariableOp%ada_cos/l2_normalize_1/ReadVariableOp2\
,batch_normalization/batchnorm/ReadVariableOp,batch_normalization/batchnorm/ReadVariableOp2`
.batch_normalization/batchnorm/ReadVariableOp_1.batch_normalization/batchnorm/ReadVariableOp_12`
.batch_normalization/batchnorm/ReadVariableOp_2.batch_normalization/batchnorm/ReadVariableOp_22d
0batch_normalization/batchnorm/mul/ReadVariableOp0batch_normalization/batchnorm/mul/ReadVariableOp2`
.batch_normalization_1/batchnorm/ReadVariableOp.batch_normalization_1/batchnorm/ReadVariableOp2d
0batch_normalization_1/batchnorm/ReadVariableOp_10batch_normalization_1/batchnorm/ReadVariableOp_12d
0batch_normalization_1/batchnorm/ReadVariableOp_20batch_normalization_1/batchnorm/ReadVariableOp_22h
2batch_normalization_1/batchnorm/mul/ReadVariableOp2batch_normalization_1/batchnorm/mul/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:R N
(
_output_shapes
:??????????
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/1
?

?
C__inference_dense_1_layer_call_and_return_conditional_losses_284137

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_283944

inputs0
!batchnorm_readvariableop_resource:	?4
%batchnorm_mul_readvariableop_resource:	?2
#batchnorm_readvariableop_1_resource:	?2
#batchnorm_readvariableop_2_resource:	?
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_284026

inputs0
!batchnorm_readvariableop_resource:	?4
%batchnorm_mul_readvariableop_resource:	?2
#batchnorm_readvariableop_1_resource:	?2
#batchnorm_readvariableop_2_resource:	?
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
b
C__inference_dropout_layer_call_and_return_conditional_losses_284342

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_1_layer_call_fn_285220

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_284073p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
b
C__inference_dropout_layer_call_and_return_conditional_losses_285174

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?*
?
A__inference_model_layer_call_and_return_conditional_losses_284270

inputs
inputs_1 
dense_284105:
??
dense_284107:	?)
batch_normalization_284110:	?)
batch_normalization_284112:	?)
batch_normalization_284114:	?)
batch_normalization_284116:	?"
dense_1_284138:
??
dense_1_284140:	?+
batch_normalization_1_284143:	?+
batch_normalization_1_284145:	?+
batch_normalization_1_284147:	?+
batch_normalization_1_284149:	?"
ada_cos_284258:
??
ada_cos_284260: 
identity??ada_cos/StatefulPartitionedCall?+ada_cos/W/Regularizer/Square/ReadVariableOp?+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_284105dense_284107*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_284104?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_284110batch_normalization_284112batch_normalization_284114batch_normalization_284116*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_283944?
dropout/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_284124?
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_284138dense_1_284140*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_284137?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_1_284143batch_normalization_1_284145batch_normalization_1_284147batch_normalization_1_284149*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_284026?
ada_cos/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0inputs_1ada_cos_284258ada_cos_284260*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_ada_cos_layer_call_and_return_conditional_losses_284257|
+ada_cos/W/Regularizer/Square/ReadVariableOpReadVariableOpada_cos_284258* 
_output_shapes
:
??*
dtype0?
ada_cos/W/Regularizer/SquareSquare3ada_cos/W/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??l
ada_cos/W/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
ada_cos/W/Regularizer/SumSum ada_cos/W/Regularizer/Square:y:0$ada_cos/W/Regularizer/Const:output:0*
T0*
_output_shapes
: `
ada_cos/W/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
ada_cos/W/Regularizer/mulMul$ada_cos/W/Regularizer/mul/x:output:0"ada_cos/W/Regularizer/Sum:output:0*
T0*
_output_shapes
: x
IdentityIdentity(ada_cos/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp ^ada_cos/StatefulPartitionedCall,^ada_cos/W/Regularizer/Square/ReadVariableOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:??????????:??????????: : : : : : : : : : : : : : 2B
ada_cos/StatefulPartitionedCallada_cos/StatefulPartitionedCall2Z
+ada_cos/W/Regularizer/Square/ReadVariableOp+ada_cos/W/Regularizer/Square/ReadVariableOp2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
&__inference_model_layer_call_fn_284502
dense_input
onehot_label
unknown:
??
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
	unknown_4:	?
	unknown_5:
??
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:	?

unknown_10:	?

unknown_11:
??

unknown_12: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_inputonehot_labelunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*+
_read_only_resource_inputs
		*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_284437p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:??????????:??????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
(
_output_shapes
:??????????
%
_user_specified_namedense_input:VR
(
_output_shapes
:??????????
&
_user_specified_nameonehot_label
?
?
4__inference_batch_normalization_layer_call_fn_285093

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_283991p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?,
?
A__inference_model_layer_call_and_return_conditional_losses_284437

inputs
inputs_1 
dense_284396:
??
dense_284398:	?)
batch_normalization_284401:	?)
batch_normalization_284403:	?)
batch_normalization_284405:	?)
batch_normalization_284407:	?"
dense_1_284411:
??
dense_1_284413:	?+
batch_normalization_1_284416:	?+
batch_normalization_1_284418:	?+
batch_normalization_1_284420:	?+
batch_normalization_1_284422:	?"
ada_cos_284425:
??
ada_cos_284427: 
identity??ada_cos/StatefulPartitionedCall?+ada_cos/W/Regularizer/Square/ReadVariableOp?+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dropout/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_284396dense_284398*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_284104?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_284401batch_normalization_284403batch_normalization_284405batch_normalization_284407*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_283991?
dropout/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_284342?
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_284411dense_1_284413*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_284137?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_1_284416batch_normalization_1_284418batch_normalization_1_284420batch_normalization_1_284422*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_284073?
ada_cos/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0inputs_1ada_cos_284425ada_cos_284427*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_ada_cos_layer_call_and_return_conditional_losses_284257|
+ada_cos/W/Regularizer/Square/ReadVariableOpReadVariableOpada_cos_284425* 
_output_shapes
:
??*
dtype0?
ada_cos/W/Regularizer/SquareSquare3ada_cos/W/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??l
ada_cos/W/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
ada_cos/W/Regularizer/SumSum ada_cos/W/Regularizer/Square:y:0$ada_cos/W/Regularizer/Const:output:0*
T0*
_output_shapes
: `
ada_cos/W/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
ada_cos/W/Regularizer/mulMul$ada_cos/W/Regularizer/mul/x:output:0"ada_cos/W/Regularizer/Sum:output:0*
T0*
_output_shapes
: x
IdentityIdentity(ada_cos/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp ^ada_cos/StatefulPartitionedCall,^ada_cos/W/Regularizer/Square/ReadVariableOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:??????????:??????????: : : : : : : : : : : : : : 2B
ada_cos/StatefulPartitionedCallada_cos/StatefulPartitionedCall2Z
+ada_cos/W/Regularizer/Square/ReadVariableOp+ada_cos/W/Regularizer/Square/ReadVariableOp2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
"__inference__traced_restore_285686
file_prefix1
assignvariableop_dense_kernel:
??,
assignvariableop_1_dense_bias:	?;
,assignvariableop_2_batch_normalization_gamma:	?:
+assignvariableop_3_batch_normalization_beta:	?A
2assignvariableop_4_batch_normalization_moving_mean:	?E
6assignvariableop_5_batch_normalization_moving_variance:	?5
!assignvariableop_6_dense_1_kernel:
??.
assignvariableop_7_dense_1_bias:	?=
.assignvariableop_8_batch_normalization_1_gamma:	?<
-assignvariableop_9_batch_normalization_1_beta:	?D
5assignvariableop_10_batch_normalization_1_moving_mean:	?H
9assignvariableop_11_batch_normalization_1_moving_variance:	?&
assignvariableop_12_variable: 1
assignvariableop_13_ada_cos_w:
??'
assignvariableop_14_adam_iter:	 )
assignvariableop_15_adam_beta_1: )
assignvariableop_16_adam_beta_2: (
assignvariableop_17_adam_decay: 0
&assignvariableop_18_adam_learning_rate: #
assignvariableop_19_total: #
assignvariableop_20_count: %
assignvariableop_21_total_1: %
assignvariableop_22_count_1: ;
'assignvariableop_23_adam_dense_kernel_m:
??4
%assignvariableop_24_adam_dense_bias_m:	?C
4assignvariableop_25_adam_batch_normalization_gamma_m:	?B
3assignvariableop_26_adam_batch_normalization_beta_m:	?=
)assignvariableop_27_adam_dense_1_kernel_m:
??6
'assignvariableop_28_adam_dense_1_bias_m:	?E
6assignvariableop_29_adam_batch_normalization_1_gamma_m:	?D
5assignvariableop_30_adam_batch_normalization_1_beta_m:	?8
$assignvariableop_31_adam_ada_cos_w_m:
??;
'assignvariableop_32_adam_dense_kernel_v:
??4
%assignvariableop_33_adam_dense_bias_v:	?C
4assignvariableop_34_adam_batch_normalization_gamma_v:	?B
3assignvariableop_35_adam_batch_normalization_beta_v:	?=
)assignvariableop_36_adam_dense_1_kernel_v:
??6
'assignvariableop_37_adam_dense_1_bias_v:	?E
6assignvariableop_38_adam_batch_normalization_1_gamma_v:	?D
5assignvariableop_39_adam_batch_normalization_1_beta_v:	?8
$assignvariableop_40_adam_ada_cos_w_v:
??
identity_42??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:**
dtype0*?
value?B?*B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-4/s/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-4/W/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-4/W/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-4/W/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:**
dtype0*g
value^B\*B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::*8
dtypes.
,2*	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp,assignvariableop_2_batch_normalization_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp+assignvariableop_3_batch_normalization_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp2assignvariableop_4_batch_normalization_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp6assignvariableop_5_batch_normalization_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_1_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp-assignvariableop_9_batch_normalization_1_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp5assignvariableop_10_batch_normalization_1_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp9assignvariableop_11_batch_normalization_1_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOpassignvariableop_12_variableIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_ada_cos_wIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_iterIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_beta_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_beta_2Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_decayIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp&assignvariableop_18_adam_learning_rateIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOpassignvariableop_19_totalIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOpassignvariableop_20_countIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOpassignvariableop_21_total_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOpassignvariableop_22_count_1Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp'assignvariableop_23_adam_dense_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp%assignvariableop_24_adam_dense_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp4assignvariableop_25_adam_batch_normalization_gamma_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp3assignvariableop_26_adam_batch_normalization_beta_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp)assignvariableop_27_adam_dense_1_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp'assignvariableop_28_adam_dense_1_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp6assignvariableop_29_adam_batch_normalization_1_gamma_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp5assignvariableop_30_adam_batch_normalization_1_beta_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp$assignvariableop_31_adam_ada_cos_w_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp'assignvariableop_32_adam_dense_kernel_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp%assignvariableop_33_adam_dense_bias_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp4assignvariableop_34_adam_batch_normalization_gamma_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp3assignvariableop_35_adam_batch_normalization_beta_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adam_dense_1_kernel_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp'assignvariableop_37_adam_dense_1_bias_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp6assignvariableop_38_adam_batch_normalization_1_gamma_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOp5assignvariableop_39_adam_batch_normalization_1_beta_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOp$assignvariableop_40_adam_ada_cos_w_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_41Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_42IdentityIdentity_41:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_42Identity_42:output:0*g
_input_shapesV
T: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402(
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
?
?
4__inference_batch_normalization_layer_call_fn_285080

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_283944p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
D
dense_input5
serving_default_dense_input:0??????????
F
onehot_label6
serving_default_onehot_label:0??????????<
ada_cos1
StatefulPartitionedCall:0??????????tensorflow/serving/predict:??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
		optimizer

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
?
axis
	gamma
beta
moving_mean
moving_variance
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses"
_tf_keras_layer
?
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)_random_generator
*__call__
*+&call_and_return_all_conditional_losses"
_tf_keras_layer
?

,kernel
-bias
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses"
_tf_keras_layer
?
4axis
	5gamma
6beta
7moving_mean
8moving_variance
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses"
_tf_keras_layer
"
_tf_keras_input_layer
?
?s
@W
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses"
_tf_keras_layer
?
Giter

Hbeta_1

Ibeta_2
	Jdecay
Klearning_ratem|m}m~m,m?-m?5m?6m?@m?v?v?v?v?,v?-v?5v?6v?@v?"
	optimizer
?
0
1
2
3
4
5
,6
-7
58
69
710
811
@12
?13"
trackable_list_wrapper
_
0
1
2
3
,4
-5
56
67
@8"
trackable_list_wrapper
'
L0"
trackable_list_wrapper
?
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics

	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
&__inference_model_layer_call_fn_284301
&__inference_model_layer_call_fn_284638
&__inference_model_layer_call_fn_284672
&__inference_model_layer_call_fn_284502?
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
A__inference_model_layer_call_and_return_conditional_losses_284824
A__inference_model_layer_call_and_return_conditional_losses_285011
A__inference_model_layer_call_and_return_conditional_losses_284547
A__inference_model_layer_call_and_return_conditional_losses_284592?
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
!__inference__wrapped_model_283920dense_inputonehot_label"?
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
,
Rserving_default"
signature_map
 :
??2dense/kernel
:?2
dense/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
&__inference_dense_layer_call_fn_285056?
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
A__inference_dense_layer_call_and_return_conditional_losses_285067?
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
 "
trackable_list_wrapper
(:&?2batch_normalization/gamma
':%?2batch_normalization/beta
0:.? (2batch_normalization/moving_mean
4:2? (2#batch_normalization/moving_variance
<
0
1
2
3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Xnon_trainable_variables

Ylayers
Zmetrics
[layer_regularization_losses
\layer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
?2?
4__inference_batch_normalization_layer_call_fn_285080
4__inference_batch_normalization_layer_call_fn_285093?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_285113
O__inference_batch_normalization_layer_call_and_return_conditional_losses_285147?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
]non_trainable_variables

^layers
_metrics
`layer_regularization_losses
alayer_metrics
%	variables
&trainable_variables
'regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
?2?
(__inference_dropout_layer_call_fn_285152
(__inference_dropout_layer_call_fn_285157?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_dropout_layer_call_and_return_conditional_losses_285162
C__inference_dropout_layer_call_and_return_conditional_losses_285174?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
": 
??2dense_1/kernel
:?2dense_1/bias
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
?2?
(__inference_dense_1_layer_call_fn_285183?
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
C__inference_dense_1_layer_call_and_return_conditional_losses_285194?
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
 "
trackable_list_wrapper
*:(?2batch_normalization_1/gamma
):'?2batch_normalization_1/beta
2:0? (2!batch_normalization_1/moving_mean
6:4? (2%batch_normalization_1/moving_variance
<
50
61
72
83"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
?
gnon_trainable_variables

hlayers
imetrics
jlayer_regularization_losses
klayer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
?2?
6__inference_batch_normalization_1_layer_call_fn_285207
6__inference_batch_normalization_1_layer_call_fn_285220?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_285240
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_285274?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
: (2Variable
:
??2	ada_cos/W
.
@0
?1"
trackable_list_wrapper
'
@0"
trackable_list_wrapper
'
L0"
trackable_list_wrapper
?
lnon_trainable_variables

mlayers
nmetrics
olayer_regularization_losses
player_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
?2?
(__inference_ada_cos_layer_call_fn_285290?
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
C__inference_ada_cos_layer_call_and_return_conditional_losses_285395?
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
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
?2?
__inference_loss_fn_0_285406?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
C
0
1
72
83
?4"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
$__inference_signature_wrapper_285047dense_inputonehot_label"?
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
 
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
.
0
1"
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
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
L0"
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	stotal
	tcount
u	variables
v	keras_api"
_tf_keras_metric
^
	wtotal
	xcount
y
_fn_kwargs
z	variables
{	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
s0
t1"
trackable_list_wrapper
-
u	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
w0
x1"
trackable_list_wrapper
-
z	variables"
_generic_user_object
%:#
??2Adam/dense/kernel/m
:?2Adam/dense/bias/m
-:+?2 Adam/batch_normalization/gamma/m
,:*?2Adam/batch_normalization/beta/m
':%
??2Adam/dense_1/kernel/m
 :?2Adam/dense_1/bias/m
/:-?2"Adam/batch_normalization_1/gamma/m
.:,?2!Adam/batch_normalization_1/beta/m
": 
??2Adam/ada_cos/W/m
%:#
??2Adam/dense/kernel/v
:?2Adam/dense/bias/v
-:+?2 Adam/batch_normalization/gamma/v
,:*?2Adam/batch_normalization/beta/v
':%
??2Adam/dense_1/kernel/v
 :?2Adam/dense_1/bias/v
/:-?2"Adam/batch_normalization_1/gamma/v
.:,?2!Adam/batch_normalization_1/beta/v
": 
??2Adam/ada_cos/W/v?
!__inference__wrapped_model_283920?,-8576@?c?`
Y?V
T?Q
&?#
dense_input??????????
'?$
onehot_label??????????
? "2?/
-
ada_cos"?
ada_cos???????????
C__inference_ada_cos_layer_call_and_return_conditional_losses_285395?@?\?Y
R?O
M?J
#? 
inputs/0??????????
#? 
inputs/1??????????
? "&?#
?
0??????????
? ?
(__inference_ada_cos_layer_call_fn_285290}@?\?Y
R?O
M?J
#? 
inputs/0??????????
#? 
inputs/1??????????
? "????????????
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_285240d85764?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_285274d78564?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
6__inference_batch_normalization_1_layer_call_fn_285207W85764?1
*?'
!?
inputs??????????
p 
? "????????????
6__inference_batch_normalization_1_layer_call_fn_285220W78564?1
*?'
!?
inputs??????????
p
? "????????????
O__inference_batch_normalization_layer_call_and_return_conditional_losses_285113d4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_285147d4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
4__inference_batch_normalization_layer_call_fn_285080W4?1
*?'
!?
inputs??????????
p 
? "????????????
4__inference_batch_normalization_layer_call_fn_285093W4?1
*?'
!?
inputs??????????
p
? "????????????
C__inference_dense_1_layer_call_and_return_conditional_losses_285194^,-0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? }
(__inference_dense_1_layer_call_fn_285183Q,-0?-
&?#
!?
inputs??????????
? "????????????
A__inference_dense_layer_call_and_return_conditional_losses_285067^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? {
&__inference_dense_layer_call_fn_285056Q0?-
&?#
!?
inputs??????????
? "????????????
C__inference_dropout_layer_call_and_return_conditional_losses_285162^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
C__inference_dropout_layer_call_and_return_conditional_losses_285174^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? }
(__inference_dropout_layer_call_fn_285152Q4?1
*?'
!?
inputs??????????
p 
? "???????????}
(__inference_dropout_layer_call_fn_285157Q4?1
*?'
!?
inputs??????????
p
? "???????????;
__inference_loss_fn_0_285406@?

? 
? "? ?
A__inference_model_layer_call_and_return_conditional_losses_284547?,-8576@?k?h
a?^
T?Q
&?#
dense_input??????????
'?$
onehot_label??????????
p 

 
? "&?#
?
0??????????
? ?
A__inference_model_layer_call_and_return_conditional_losses_284592?,-7856@?k?h
a?^
T?Q
&?#
dense_input??????????
'?$
onehot_label??????????
p

 
? "&?#
?
0??????????
? ?
A__inference_model_layer_call_and_return_conditional_losses_284824?,-8576@?d?a
Z?W
M?J
#? 
inputs/0??????????
#? 
inputs/1??????????
p 

 
? "&?#
?
0??????????
? ?
A__inference_model_layer_call_and_return_conditional_losses_285011?,-7856@?d?a
Z?W
M?J
#? 
inputs/0??????????
#? 
inputs/1??????????
p

 
? "&?#
?
0??????????
? ?
&__inference_model_layer_call_fn_284301?,-8576@?k?h
a?^
T?Q
&?#
dense_input??????????
'?$
onehot_label??????????
p 

 
? "????????????
&__inference_model_layer_call_fn_284502?,-7856@?k?h
a?^
T?Q
&?#
dense_input??????????
'?$
onehot_label??????????
p

 
? "????????????
&__inference_model_layer_call_fn_284638?,-8576@?d?a
Z?W
M?J
#? 
inputs/0??????????
#? 
inputs/1??????????
p 

 
? "????????????
&__inference_model_layer_call_fn_284672?,-7856@?d?a
Z?W
M?J
#? 
inputs/0??????????
#? 
inputs/1??????????
p

 
? "????????????
$__inference_signature_wrapper_285047?,-8576@?}?z
? 
s?p
5
dense_input&?#
dense_input??????????
7
onehot_label'?$
onehot_label??????????"2?/
-
ada_cos"?
ada_cos??????????