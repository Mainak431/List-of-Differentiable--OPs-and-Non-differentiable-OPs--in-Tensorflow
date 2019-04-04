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

"DebugGradientIdentity" \
"DebugGradientRefIdentity" \
name \
"If" \
"Roll" \
"Conv2DBackpropInput" \
"Conv2DBackpropFilter" \
"Conv3D" \
"Conv3DBackpropInputV2" \
"Conv3DBackpropFilterV2" \
"AvgPool3D" \
"AvgPool3DGrad" \
"MaxPool3D" \
"MaxPool3DGrad" \
"MaxPool3DGradGrad" \
"Softmax" \
"LogSoftmax" \
"BiasAdd" \
"BiasAddGrad" \
"BiasAddV1" \
"Relu" \
"EluGrad" \
"SeluGrad" \
"Relu6" \
"Relu6Grad" \
"LeakyRelu" \
"LeakyReluGrad" \
"Elu" \
"Selu" \
"Softplus" \
"SoftplusGrad" \
"Softsign" \
"ReluGrad" \
"SoftmaxCrossEntropyWithLogits" \
"SparseSoftmaxCrossEntropyWithLogits" \
"Conv2D" \
"DepthwiseConv2dNative" \
"Dilation2D" \
"LRN" \
"AvgPool" \
"AvgPoolGrad" \
"MaxPool" \
"MaxPoolV2" \
"MaxPoolWithArgmax" \
"MaxPoolGrad" \
"MaxPoolGradV2" \
"MaxPoolGradGrad" \
"FractionalMaxPool" \
"FractionalAvgPool" \
"BatchNormWithGlobalNormalization" \
"FusedBatchNorm" \
"FusedBatchNormV2" \
"FusedBatchNormGrad" \
"FusedBatchNormGradV2" \
"L2Loss" \
"TopK" \
"TopKV2" \
"NthElement" \
'NcclAllReduce' \
'NcclReduce' \
'NcclBroadcast' \
"ResizeNearestNeighbor" \
"ResizeBilinear" \
"ResizeBicubic" \
"CropAndResize" \
"ClipByValue" \
"ReadVariableOp" \
"ResourceGather" \
"CudnnRNN" \
"CudnnRNNV2" \
"Pack" \
"Unpack" \
"Concat" \
"ConcatV2" \
"Slice" \
"StridedSlice" \
"StridedSliceGrad" \
"Split" \
"SplitV" \
"Diag" \
"DiagPart" \
"MatrixDiag" \
"MatrixDiagPart" \
"MatrixSetDiag" \
"MatrixBandPart" \
"Fill" \
"PreventGradient" \
"Gather" \
"GatherV2" \
"GatherNd" \
"CheckNumerics" \
"PlaceholderWithDefault" \
"Identity" \
"RefIdentity" \
"IdentityN" \
"Reshape" \
"ExpandDims" \
"Squeeze" \
"Transpose" \
"ConjugateTranspose" \
"Tile" \
"Pad" \
"PadV2" \
"ReverseSequence" \
"Reverse" \
"ReverseV2" \
"SpaceToBatch" \
"SpaceToBatchND" \
"BatchToSpace" \
"BatchToSpaceND" \
"SpaceToDepth" \
"DepthToSpace" \
"MirrorPad" \
"MirrorPadGrad" \
"QuantizeAndDequantize" \
"QuantizeAndDequantizeV2" \
"QuantizeAndDequantizeV3" \
"ExtractImagePatches" \
"ScatterNd" \
"TensorScatterUpdate" \
"TensorScatterAdd" \
"TensorScatterSub" \
"ScatterNdNonAliasingAdd" \
"BroadcastTo" \
"CTCLoss" \
"TensorArrayRead" \
"TensorArrayReadV2" \
"TensorArrayReadV3" \
"TensorArrayWrite" \
"TensorArrayWriteV2" \
"TensorArrayWriteV3" \
"TensorArrayGather" \
"TensorArrayGatherV2" \
"TensorArrayGatherV3" \
"TensorArrayScatter" \
"TensorArrayScatterV2" \
"TensorArrayScatterV3" \
"TensorArrayConcat" \
"TensorArrayConcatV2" \
"TensorArrayConcatV3" \
"TensorArraySplit" \
"TensorArraySplitV2" \
"TensorArraySplitV3" \
"Print" \
name \
'EnsureShape' \
"OptionalFromValue" \
"OptionalGetValue" \
"SparseReorder" \
"SparseAdd" \
"SparseTensorDenseAdd" \
"SparseReduceSum" \
"SparseSlice" \
"SparseTensorDenseMatMul" \
"SparseDenseCwiseAdd" \
"SparseDenseCwiseMul" \
"SparseDenseCwiseDiv" \
"SparseSoftmax" \
"SparseSparseMaximum" \
"SparseSparseMinimum" \
"SparseFillEmptyRows" \
"AccumulateNV2" \
"RandomGamma" \
"MatrixInverse" \
"MatrixDeterminant" \
"MatrixSquareRoot" \
"LogMatrixDeterminant" \
"Cholesky" \
"Qr" \
"MatrixSolve" \
"MatrixSolveLs" \
"MatrixTriangularSolve" \
"SelfAdjointEigV2" \
"Svd" \
"TensorListPushBack" \
"TensorListPopBack" \
"TensorListStack" \
"TensorListConcat" \
"TensorListSplit" \
"TensorListFromTensor" \
"TensorListGetItem" \
"TensorListSetItem" \
"TensorListGather" \
"TensorListScatter" \
"EagerPyFunc" \
"FakeQuantWithMinMaxArgs" \
"FakeQuantWithMinMaxVars" \
"FakeQuantWithMinMaxVarsPerChannel" \
"DynamicPartition" \
"DynamicStitch" \
"ParallelDynamicStitch" \
"While" \
"Switch" \
"RefSwitch" \
"Merge" \
"RefMerge" \
"Exit" \
"RefExit" \
"NextIteration" \
"RefNextIteration" \
"Enter" \
"RefEnter" \
"LoopCond" \
"ArgMax" \
"ArgMin" \
"Sum" \
"Max" \
"Min" \
"Mean" \
"Prod" \
"SegmentSum" \
"SegmentMean" \
"SparseSegmentSum" \
"SparseSegmentSumWithNumSegments" \
"SparseSegmentMean" \
"SparseSegmentMeanWithNumSegments" \
"SparseSegmentSqrtN" \
"SparseSegmentSqrtNWithNumSegments" \
"SegmentMin" \
"SegmentMax" \
"UnsortedSegmentSum" \
"UnsortedSegmentMax" \
"UnsortedSegmentMin" \
"UnsortedSegmentProd" \
"Abs" \
"Neg" \
"Inv" \
"Reciprocal" \
"InvGrad" \
"ReciprocalGrad" \
"Square" \
"Sqrt" \
"SqrtGrad" \
"Rsqrt" \
"RsqrtGrad" \
"Exp" \
"Expm1" \
"Log" \
"Log1p" \
"Xlogy" \
"Xdivy" \
"Sinh" \
"Cosh" \
"Tanh" \
"Asinh" \
"Acosh" \
"Atanh" \
"TanhGrad" \
"Erf" \
"Erfc" \
"Lgamma" \
"Digamma" \
"BesselI0e" \
"BesselI1e" \
"Igamma" \
"Igammac" \
"Betainc" \
"Zeta" \
"Polygamma" \
"Sigmoid" \
"SigmoidGrad" \
"Sign" \
"Sin" \
"Cos" \
"Tan" \
"Asin" \
"Acos" \
"Atan" \
"Atan2" \
"AddN" \
"Add" \
"Sub" \
"Mul" \
"Div" \
"FloorDiv" \
"FloorMod" \
"TruncateDiv" \
"RealDiv" \
"DivNoNan" \
"Pow" \
"Maximum" \
"Minimum" \
"SquaredDifference" \
"Select" \
"MatMul" \
"SparseMatMul" \
"Floor" \
"Ceil" \
"Round" \
"Rint" \
"BatchMatMul" \
"Complex" \
"Real" \
"Imag" \
"Angle" \
"Conj" \
"ComplexAbs" \
"Cast" \
"Cross" \
"Cumsum" \
"Cumprod" \
"FFT" \
"IFFT" \
"FFT2D" \
"IFFT2D" \
"FFT3D" \
"IFFT3D" \
"RFFT" \
"IRFFT" \
"RFFT2D" \
"IRFFT2D" \
"RaggedTensorToSparse" \
"XlaClusterOutput" \
'GDNLowerBound' \
"ImageProjectiveTransformV2" \
'FoldFusedBatchNormGrad' \
"PeriodicResample" \
"GRUBlockCell" \
"LSTMBlockCell" \
"BlockLSTM" \
'TPUEmbeddingActivations' \
"AllToAll" \
"CollectivePermute" \
"CrossReplicaSum" \
"Resampler" \
'RoutingFunction' \
'StochasticHardRoutingFunction' \
'KFeatureRoutingFunction' \
"Batch" \
"Unbatch" \



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

