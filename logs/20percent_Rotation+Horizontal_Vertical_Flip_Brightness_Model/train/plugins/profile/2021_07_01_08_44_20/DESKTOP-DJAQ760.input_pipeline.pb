	^?I?T@^?I?T@!^?I?T@	?1X(?G@?1X(?G@!?1X(?G@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$^?I?T@ŏ1w-!??AL7?A`?E@Y??N@?C@*	3333P!"A2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator?L??W?@!?/J?ZW@)?L??W?@1?/J?ZW@:Preprocessing2P
Iterator::Model::Prefetchj?q???C@!?<??$S@)j?q???C@1?<??$S@:Preprocessing2F
Iterator::Model??_vO?C@!?Hcv?U@)???߾??1?\a &?b?:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap??_v?W?@!u˙??ZW@)a2U0*?c?1?Z???y:?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 47.2% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9?1X(?G@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	ŏ1w-!??ŏ1w-!??!ŏ1w-!??      ??!       "      ??!       *      ??!       2	L7?A`?E@L7?A`?E@!L7?A`?E@:      ??!       B      ??!       J	??N@?C@??N@?C@!??N@?C@R      ??!       Z	??N@?C@??N@?C@!??N@?C@JCPU_ONLYY?1X(?G@b 