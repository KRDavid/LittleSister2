?	&S???R@&S???R@!&S???R@	?,?}?D@?,?}?D@!?,?}?D@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$&S???R@I??&??A?6?[?F@Y5^?Ir>@*	43333!A2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator????K??@!s????W@)????K??@1s????W@:Preprocessing2P
Iterator::Model::Prefetch*??Dh>@!Lk@є@)*??Dh>@1Lk@є@:Preprocessing2F
Iterator::Model?Q?k>@!?X?p\?@)y?&1???1;?j/?Yd?:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMapݵ?|P??@!v??8??W@)HP?s?b?1?{g?;:?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 40.2% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9?,?}?D@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	I??&??I??&??!I??&??      ??!       "      ??!       *      ??!       2	?6?[?F@?6?[?F@!?6?[?F@:      ??!       B      ??!       J	5^?Ir>@5^?Ir>@!5^?Ir>@R      ??!       Z	5^?Ir>@5^?Ir>@!5^?Ir>@JCPU_ONLYY?,?}?D@b Y      Y@q?3V?]@"?
host?Your program is HIGHLY input-bound because 40.2% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"CPU: B 