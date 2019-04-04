# List-of-Differentiable-Ops-and-Non-differentiable-Ops-in-Tensorflow


This list is based on tensorflow version 1.13.

There are many Differentiable and many Non-Differentiable Ops present in Tensorflow currently. Look at the lists if you want the lists differnetly. 

This list is based on the tag :

Differentiable - ops.RegisterGradient 
& 
Non-Differentiable- ops.NotDifferentiable

# Differentiable List - 
https://github.com/Mainak431/List-of-Differentiable--OPs-and-Non-differentiable-OPs--in-Tensorflow/blob/master/DIFFERENTIABLE%20LIST.txt

# Non-Differentiable List -
https://github.com/Mainak431/List-of-Differentiable--OPs-and-Non-differentiable-OPs--in-Tensorflow/blob/master/NonDifferentiable.txt

# The code which created the lists are given 
https://github.com/Mainak431/List-of-Differentiable--OPs-and-Non-differentiable-OPs--in-Tensorflow/blob/master/extract_all_ops.py

# DIFFERENTIABLE OPS:

"DebugGradientIdentity" &nbsp;
"DebugGradientRefIdentity" &nbsp;
"If" &nbsp;
"Roll" &nbsp;
"Conv2DBackpropInput" &nbsp;
"Conv2DBackpropFilter" &nbsp;
"Conv3D" &nbsp;
"Conv3DBackpropInputV2" &nbsp;
"Conv3DBackpropFilterV2" &nbsp;
"AvgPool3D" &nbsp;
"AvgPool3DGrad" &nbsp;
"MaxPool3D" &nbsp;
"MaxPool3DGrad" &nbsp;
"MaxPool3DGradGrad" &nbsp;
"Softmax" &nbsp;
"LogSoftmax" &nbsp;
"BiasAdd" &nbsp;
"BiasAddGrad" &nbsp;
"BiasAddV1" &nbsp;
"Relu" &nbsp;
"EluGrad" &nbsp;
"SeluGrad" &nbsp;
"Relu6" &nbsp;
"Relu6Grad" &nbsp;
"LeakyRelu" &nbsp;
"LeakyReluGrad" &nbsp;
"Elu" &nbsp;
"Selu" &nbsp;
"Softplus" &nbsp;
"SoftplusGrad" &nbsp;
"Softsign" &nbsp;
"ReluGrad" &nbsp;
"SoftmaxCrossEntropyWithLogits" &nbsp;
"SparseSoftmaxCrossEntropyWithLogits" &nbsp;
"Conv2D" &nbsp;
"DepthwiseConv2dNative" &nbsp;
"Dilation2D" &nbsp;
"LRN" &nbsp;
"AvgPool" &nbsp;
"AvgPoolGrad" &nbsp;
"MaxPool" &nbsp;
"MaxPoolV2" &nbsp;
"MaxPoolWithArgmax" &nbsp;
"MaxPoolGrad" &nbsp;
"MaxPoolGradV2" &nbsp;
"MaxPoolGradGrad" &nbsp;
"FractionalMaxPool" &nbsp;
"FractionalAvgPool" &nbsp;
"BatchNormWithGlobalNormalization" &nbsp;
"FusedBatchNorm" &nbsp;
"FusedBatchNormV2" &nbsp;
"FusedBatchNormGrad" &nbsp;
"FusedBatchNormGradV2" &nbsp;
"L2Loss" &nbsp;
"TopK" &nbsp;
"TopKV2" &nbsp;
"NthElement" &nbsp;
'NcclAllReduce' &nbsp;
'NcclReduce' &nbsp;
'NcclBroadcast' &nbsp;
"ResizeNearestNeighbor" &nbsp;
"ResizeBilinear" &nbsp;
"ResizeBicubic" &nbsp;
"CropAndResize" &nbsp;
"ClipByValue" &nbsp;
"ReadVariableOp" &nbsp;
"ResourceGather" &nbsp;
"CudnnRNN" &nbsp;
"CudnnRNNV2" &nbsp;
"Pack" &nbsp;
"Unpack" &nbsp;
"Concat" &nbsp;
"ConcatV2" &nbsp;
"Slice" &nbsp;
"StridedSlice" &nbsp;
"StridedSliceGrad" &nbsp;
"Split" &nbsp;
"SplitV" &nbsp;
"Diag" &nbsp;
"DiagPart" &nbsp;
"MatrixDiag" &nbsp;
"MatrixDiagPart" &nbsp;
"MatrixSetDiag" &nbsp;
"MatrixBandPart" &nbsp;
"Fill" &nbsp;
"PreventGradient" &nbsp;
"Gather" &nbsp;
"GatherV2" &nbsp;
"GatherNd" &nbsp;
"CheckNumerics" &nbsp;
"PlaceholderWithDefault" &nbsp;
"Identity" &nbsp;
"RefIdentity" &nbsp;
"IdentityN" &nbsp;
"Reshape" &nbsp;
"ExpandDims" &nbsp;
"Squeeze" &nbsp;
"Transpose" &nbsp;
"ConjugateTranspose" &nbsp;
"Tile" &nbsp;
"Pad" &nbsp;
"PadV2" &nbsp;
"ReverseSequence" &nbsp;
"Reverse" &nbsp;
"ReverseV2" &nbsp;
"SpaceToBatch" &nbsp;
"SpaceToBatchND" &nbsp;
"BatchToSpace" &nbsp;
"BatchToSpaceND" &nbsp;
"SpaceToDepth" &nbsp;
"DepthToSpace" &nbsp;
"MirrorPad" &nbsp;
"MirrorPadGrad" &nbsp;
"QuantizeAndDequantize" &nbsp;
"QuantizeAndDequantizeV2" &nbsp;
"QuantizeAndDequantizeV3" &nbsp;
"ExtractImagePatches" &nbsp;
"ScatterNd" &nbsp;
"TensorScatterUpdate" &nbsp;
"TensorScatterAdd" &nbsp;
"TensorScatterSub" &nbsp;
"ScatterNdNonAliasingAdd" &nbsp;
"BroadcastTo" &nbsp;
"CTCLoss" &nbsp;
"TensorArrayRead" &nbsp;
"TensorArrayReadV2" &nbsp;
"TensorArrayReadV3" &nbsp;
"TensorArrayWrite" &nbsp;
"TensorArrayWriteV2" &nbsp;
"TensorArrayWriteV3" &nbsp;
"TensorArrayGather" &nbsp;
"TensorArrayGatherV2" &nbsp;
"TensorArrayGatherV3" &nbsp;
"TensorArrayScatter" &nbsp;
"TensorArrayScatterV2" &nbsp;
"TensorArrayScatterV3" &nbsp;
"TensorArrayConcat" &nbsp;
"TensorArrayConcatV2" &nbsp;
"TensorArrayConcatV3" &nbsp;
"TensorArraySplit" &nbsp;
"TensorArraySplitV2" &nbsp;
"TensorArraySplitV3" &nbsp;
"Print" &nbsp;
name &nbsp;
'EnsureShape' &nbsp;
"OptionalFromValue" &nbsp;
"OptionalGetValue" &nbsp;
"SparseReorder" &nbsp;
"SparseAdd" &nbsp;
"SparseTensorDenseAdd" &nbsp;
"SparseReduceSum" &nbsp;
"SparseSlice" &nbsp;
"SparseTensorDenseMatMul" &nbsp;
"SparseDenseCwiseAdd" &nbsp;
"SparseDenseCwiseMul" &nbsp;
"SparseDenseCwiseDiv" &nbsp;
"SparseSoftmax" &nbsp;
"SparseSparseMaximum" &nbsp;
"SparseSparseMinimum" &nbsp;
"SparseFillEmptyRows" &nbsp;
"AccumulateNV2" &nbsp;
"RandomGamma" &nbsp;
"MatrixInverse" &nbsp;
"MatrixDeterminant" &nbsp;
"MatrixSquareRoot" &nbsp;
"LogMatrixDeterminant" &nbsp;
"Cholesky" &nbsp;
"Qr" &nbsp;
"MatrixSolve" &nbsp;
"MatrixSolveLs" &nbsp;
"MatrixTriangularSolve" &nbsp;
"SelfAdjointEigV2" &nbsp;
"Svd" &nbsp;
"TensorListPushBack" &nbsp;
"TensorListPopBack" &nbsp;
"TensorListStack" &nbsp;
"TensorListConcat" &nbsp;
"TensorListSplit" &nbsp;
"TensorListFromTensor" &nbsp;
"TensorListGetItem" &nbsp;
"TensorListSetItem" &nbsp;
"TensorListGather" &nbsp;
"TensorListScatter" &nbsp;
"EagerPyFunc" &nbsp;
"FakeQuantWithMinMaxArgs" &nbsp;
"FakeQuantWithMinMaxVars" &nbsp;
"FakeQuantWithMinMaxVarsPerChannel" &nbsp;
"DynamicPartition" &nbsp;
"DynamicStitch" &nbsp;
"ParallelDynamicStitch" &nbsp;
"While" &nbsp;
"Switch" &nbsp;
"RefSwitch" &nbsp;
"Merge" &nbsp;
"RefMerge" &nbsp;
"Exit" &nbsp;
"RefExit" &nbsp;
"NextIteration" &nbsp;
"RefNextIteration" &nbsp;
"Enter" &nbsp;
"RefEnter" &nbsp;
"LoopCond" &nbsp;
"ArgMax" &nbsp;
"ArgMin" &nbsp;
"Sum" &nbsp;
"Max" &nbsp;
"Min" &nbsp;
"Mean" &nbsp;
"Prod" &nbsp;
"SegmentSum" &nbsp;
"SegmentMean" &nbsp;
"SparseSegmentSum" &nbsp;
"SparseSegmentSumWithNumSegments" &nbsp;
"SparseSegmentMean" &nbsp;
"SparseSegmentMeanWithNumSegments" &nbsp;
"SparseSegmentSqrtN" &nbsp;
"SparseSegmentSqrtNWithNumSegments" &nbsp;
"SegmentMin" &nbsp;
"SegmentMax" &nbsp;
"UnsortedSegmentSum" &nbsp;
"UnsortedSegmentMax" &nbsp;
"UnsortedSegmentMin" &nbsp;
"UnsortedSegmentProd" &nbsp;
"Abs" &nbsp;
"Neg" &nbsp;
"Inv" &nbsp;
"Reciprocal" &nbsp;
"InvGrad" &nbsp;
"ReciprocalGrad" &nbsp;
"Square" &nbsp;
"Sqrt" &nbsp;
"SqrtGrad" &nbsp;
"Rsqrt" &nbsp;
"RsqrtGrad" &nbsp;
"Exp" &nbsp;
"Expm1" &nbsp;
"Log" &nbsp;
"Log1p" &nbsp;
"Xlogy" &nbsp;
"Xdivy" &nbsp;
"Sinh" &nbsp;
"Cosh" &nbsp;
"Tanh" &nbsp;
"Asinh" &nbsp;
"Acosh" &nbsp;
"Atanh" &nbsp;
"TanhGrad" &nbsp;
"Erf" &nbsp;
"Erfc" &nbsp;
"Lgamma" &nbsp;
"Digamma" &nbsp;
"BesselI0e" &nbsp;
"BesselI1e" &nbsp;
"Igamma" &nbsp;
"Igammac" &nbsp;
"Betainc" &nbsp;
"Zeta" &nbsp;
"Polygamma" &nbsp;
"Sigmoid" &nbsp;
"SigmoidGrad" &nbsp;
"Sign" &nbsp;
"Sin" &nbsp;
"Cos" &nbsp;
"Tan" &nbsp;
"Asin" &nbsp;
"Acos" &nbsp;
"Atan" &nbsp;
"Atan2" &nbsp;
"AddN" &nbsp;
"Add" &nbsp;
"Sub" &nbsp;
"Mul" &nbsp;
"Div" &nbsp;
"FloorDiv" &nbsp;
"FloorMod" &nbsp;
"TruncateDiv" &nbsp;
"RealDiv" &nbsp;
"DivNoNan" &nbsp;
"Pow" &nbsp;
"Maximum" &nbsp;
"Minimum" &nbsp;
"SquaredDifference" &nbsp;
"Select" &nbsp;
"MatMul" &nbsp;
"SparseMatMul" &nbsp;
"Floor" &nbsp;
"Ceil" &nbsp;
"Round" &nbsp;
"Rint" &nbsp;
"BatchMatMul" &nbsp;
"Complex" &nbsp;
"Real" &nbsp;
"Imag" &nbsp;
"Angle" &nbsp;
"Conj" &nbsp;
"ComplexAbs" &nbsp;
"Cast" &nbsp;
"Cross" &nbsp;
"Cumsum" &nbsp;
"Cumprod" &nbsp;
"FFT" &nbsp;
"IFFT" &nbsp;
"FFT2D" &nbsp;
"IFFT2D" &nbsp;
"FFT3D" &nbsp;
"IFFT3D" &nbsp;
"RFFT" &nbsp;
"IRFFT" &nbsp;
"RFFT2D" &nbsp;
"IRFFT2D" &nbsp;
"RaggedTensorToSparse" &nbsp;
"XlaClusterOutput" &nbsp;
'GDNLowerBound' &nbsp;
"ImageProjectiveTransformV2" &nbsp;
'FoldFusedBatchNormGrad' &nbsp;
"PeriodicResample" &nbsp;
"GRUBlockCell" &nbsp;
"LSTMBlockCell" &nbsp;
"BlockLSTM" &nbsp;
'TPUEmbeddingActivations' &nbsp;
"AllToAll" &nbsp;
"CollectivePermute" &nbsp;
"CrossReplicaSum" &nbsp;
"Resampler" &nbsp;
'RoutingFunction' &nbsp;
'StochasticHardRoutingFunction' &nbsp;
'KFeatureRoutingFunction' &nbsp;
"Batch" &nbsp;
"Unbatch" &nbsp;



# NON DIFFERENTIABLE OPS

"ReaderRead" 
"ReaderReadUpTo" 
"ReaderNumRecordsProduced" 
"ReaderNumWorkUnitsCompleted" 
"ReaderSerializeState" 
"ReaderRestoreState" 
"ReaderReset" 
"WholeFileReader" 
"TextLineReader" 
"FixedLengthRecordReader" 
"TFRecordReader" 
"LMDBReader" 
"IdentityReader" 
"BitwiseAnd" 
"BitwiseOr" 
"BitwiseXor" 
"Invert" 
"PopulationCount" 
"LeftShift" 
"RightShift" 
'RandomCrop' 
'RGBToHSV' 
'HSVToRGB' 
'DrawBoundingBoxes' 
'SampleDistortedBoundingBox' 
'SampleDistortedBoundingBoxV2' 
'ExtractGlimpse' 
'NonMaxSuppression' 
'NonMaxSuppressionV2' 
'NonMaxSuppressionWithOverlaps' 
"VarIsInitializedOp" 
"VariableShape" 
"ConcatOffset" 
"Const" 
"EditDistance" 
"ZerosLike" 
"OnesLike" 
"StopGradient" 
"InvertPermutation" 
"Shape" 
"ShapeN" 
"Rank" 
"Size" 
"BroadcastGradientArgs" 
"OneHot" 
"SetSize" 
"DenseToDenseSetOperation" 
"DenseToSparseSetOperation" 
"SparseToSparseSetOperation" 
"CTCGreedyDecoder" 
"CTCBeamSearchDecoder" 
"TensorArray" 
"TensorArrayGrad" 
"TensorArraySize" 
"TensorArrayClose" 
"TensorArrayV2" 
"TensorArrayGradV2" 
"TensorArraySizeV2" 
"TensorArrayCloseV2" 
"TensorArrayV3" 
"TensorArrayGradV3" 
"TensorArrayGradWithShape" 
"TensorArraySizeV3" 
"TensorArrayCloseV3" 
"RegexReplace" 
"StringToHashBucket" 
"StringToHashBucketFast" 
"StringToHashBucketStrong" 
"ReduceJoin" 
"StringJoin" 
"StringSplit" 
"AsString" 
"EncodeBase64" 
"DecodeBase64" 
"HistogramSummary" 
"ImageSummary" 
"AudioSummary" 
"AudioSummaryV2" 
"MergeSummary" 
"ScalarSummary" 
"TensorSummary" 
"TensorSummaryV2" 
"Timestamp" 
"Assign" 
"AssignAdd" 
"AssignSub" 
"ScatterAdd" 
"ScatterSub" 
"ScatterMul" 
"ScatterDiv" 
"ScatterNdUpdate" 
"ScatterNdAdd" 
"ScatterNdSub" 
"ScatterNdMul" 
"ScatterNdDiv" 
"SparseAddGrad" 
"SparseConcat" 
"SparseToDense" 
"RandomStandardNormal" 
"ParameterizedTruncatedNormal" 
"TruncatedNormal" 
"RandomUniform" 
"Multinomial" 
"LookupTableFind" 
"LookupTableFindV2" 
"LookupTableInsert" 
"LookupTableInsertV2" 
"LookupTableSize" 
"LookupTableSizeV2" 
"HashTable" 
"HashTableV2" 
"InitializeTable" 
"InitializeTableV2" 
"InitializeTableFromTextFile" 
"InitializeTableFromTextFileV2" 
"MutableDenseHashTable" 
"MutableDenseHashTableV2" 
"MutableHashTable" 
"MutableHashTableV2" 
"MutableHashTableOfTensors" 
"MutableHashTableOfTensorsV2" 
"DecodeRaw" 
"ParseTensor" 
"SerializeTensor" 
"StringToNumber" 
"StatelessMultinomial" 
"StatelessRandomNormal" 
"StatelessRandomUniform" 
"StatelessRandomUniformInt" 
"StatelessTruncatedNormal" 
"TensorListConcatLists" 
"TensorListElementShape" 
"TensorListLength" 
"TensorListPushBackBatch" 
"PyFunc" 
"PyFuncStateless" 
"Queue" 
"QueueEnqueue" 
"QueueEnqueueMany" 
"QueueDequeue" 
"QueueDequeueMany" 
"QueueDequeueUpTo" 
"QueueClose" 
"QueueSize" 
"Stack" 
"StackPush" 
"StackPop" 
"StackClose" 
"GetSessionHandle" 
"GetSessionHandleV2" 
"GetSessionTensor" 
"DeleteSessionTensor" 
"SdcaFprint" 
"SdcaOptimizer" 
"SdcaOptimizerV2" 
"SdcaShrinkL1" 
"Less" 
"LessEqual" 
"Greater" 
"GreaterEqual" 
"Equal" 
"ApproximateEqual" 
"NotEqual" 
"LogicalAnd" 
"LogicalOr" 
"LogicalNot" 
"Range" 
"LinSpace" 
"GenerateVocabRemapping" 
"LoadAndRemapMatrix" 
"ReduceDataset" 
"SparseFeatureCross" 
"SparseFeatureCrossV2" 
"BipartiteMatch" 
"ImageConnectedComponents" 
"SingleImageRandomDotStereograms" 
"BigQueryReader" 
"RemoteFusedGraphExecute" 
"SkipGramGenerateCandidates" 
"Rpc" 
"TryRpc" 
"DecodeLibSVM" 
"TreeEnsembleVariable" 
"TreeEnsembleSerialize" 
"TreeEnsembleDeserialize" 
"ResamplerGrad" 
"FertileStatsVariable" 
"FertileStatsSerialize" 
"FertileStatsDeserialize" 
"GrowTreeV4" 
"ProcessInputV4" 
"FinalizeTree" 
"TreeVariable" 
"TreeSerialize" 
"TreeDeserialize" 
"TreeSize" 
"TreePredictionsV4" 
"FeatureUsageCounts" 
'HardRoutingFunction' 
'RoutingGradient' 
'KFeatureDataGradient' 
'KFeatureRoutingGradient' 
'KFeatureWeightGradient' 
'UnpackPath' 
"DecodeProtoV2" 
"EncodeProto" 
'DecodeAudio' 
'EncodeAudio' 
'DecodeVideo' 
"HyperplaneLSHProbes" 

