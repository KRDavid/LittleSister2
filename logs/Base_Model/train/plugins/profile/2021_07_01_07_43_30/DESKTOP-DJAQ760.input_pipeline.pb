	a2U0*9F@a2U0*9F@!a2U0*9F@	>??U????>??U????!>??U????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$a2U0*9F@-!?lV??A@?߾$F@Y?C??????*	gffff*?@2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator[Ӽ??K@!?% /4?X@)[Ӽ??K@1?% /4?X@:Preprocessing2F
Iterator::ModelK?=?U??!???[)??)??_?L??1??"?O$??:Preprocessing2P
Iterator::Model::Prefetchn????!????	??)n????1????	??:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap??e?c?K@!?t?z?X@)a2U0*?c?1?]?]?q?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9>??U????#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	-!?lV??-!?lV??!-!?lV??      ??!       "      ??!       *      ??!       2	@?߾$F@@?߾$F@!@?߾$F@:      ??!       B      ??!       J	?C???????C??????!?C??????R      ??!       Z	?C???????C??????!?C??????JCPU_ONLYY>??U????b 