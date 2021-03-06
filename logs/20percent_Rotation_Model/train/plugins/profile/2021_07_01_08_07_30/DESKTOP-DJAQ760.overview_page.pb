?	??a???P@??a???P@!??a???P@	n@???	D@n@???	D@!n@???	D@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??a???P@???S????A-C??&D@Y???H;@*	   .A2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator?_vOx@!???8?\W@)?_vOx@1???8?\W@:Preprocessing2P
Iterator::Model::Prefetch-????;@!@?D<4@)-????;@1@?D<4@:Preprocessing2F
Iterator::Model????	;@!-(3?7@)?+e?X??18e???f?:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap?`TR'x@!~?|??\W@)/n??b?1x?)!jyA?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 40.1% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9m@???	D@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???S???????S????!???S????      ??!       "      ??!       *      ??!       2	-C??&D@-C??&D@!-C??&D@:      ??!       B      ??!       J	???H;@???H;@!???H;@R      ??!       Z	???H;@???H;@!???H;@JCPU_ONLYYm@???	D@b Y      Y@q8??`??%@"?
host?Your program is HIGHLY input-bound because 40.1% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?10.8922% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 