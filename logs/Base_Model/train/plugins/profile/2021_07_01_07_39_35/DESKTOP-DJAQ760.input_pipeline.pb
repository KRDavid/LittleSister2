	sh?????@sh?????@!sh?????@	??+l%o6@??+l%o6@!??+l%o6@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$sh?????@|a2U0*??A0*?З?@Y㥛? ?u@*	?????A2P
Iterator::Model::Prefetch?i?q??u@!?M?[??X@)?i?q??u@1?M?[??X@:Preprocessing2F
Iterator::Model A?ĉu@!      Y@)??ܵ?|??12g??)s?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 22.4% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9??+l%o6@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	|a2U0*??|a2U0*??!|a2U0*??      ??!       "      ??!       *      ??!       2	0*?З?@0*?З?@!0*?З?@:      ??!       B      ??!       J	㥛? ?u@㥛? ?u@!㥛? ?u@R      ??!       Z	㥛? ?u@㥛? ?u@!㥛? ?u@JCPU_ONLYY??+l%o6@b 