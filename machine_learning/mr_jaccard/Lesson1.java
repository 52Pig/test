package com.nowcoder.course;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.MultipleInputs;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

public class Lesson1 extends Configured implements Tool {
  @Override
  public int run(String[] args) throws Exception {
    // 1.step 1
    Job job1 = Job.getInstance(getConf(), "step1");
    job1.setJarByClass(Lesson1.class);
    job1.setMapperClass(Mapper1.class);
    job1.setReducerClass(Reducer1.class);
    job1.setOutputKeyClass(Text.class);
    job1.setOutputValueClass(IntWritable.class);
    
    FileInputFormat.addInputPath(job1, new Path(args[0]));
    FileOutputFormat.setOutputPath(job1, new Path(args[1] + "/step1"));
    job1.waitForCompletion(true);
    
    // 2.step 2
    Job job2 = Job.getInstance(getConf(), "step2");
    job2.setJarByClass(Lesson1.class);
    job2.setReducerClass(Reducer2.class);
    job2.setOutputKeyClass(IntWritable.class);
    job2.setOutputValueClass(Text.class);
    
    MultipleInputs.addInputPath(job2, new Path(args[1] + "/step1"), TextInputFormat.class, Mapper21.class);
    MultipleInputs.addInputPath(job2, new Path(args[0]), TextInputFormat.class, Mapper22.class);
    FileOutputFormat.setOutputPath(job2, new Path(args[1] + "/step2"));
    job2.waitForCompletion(true);
    // 3.step 3
    Job job3 = Job.getInstance(getConf(), "step3");
    job3.setJarByClass(Lesson1.class);
    job3.setReducerClass(Reducer3.class);
    job3.setOutputKeyClass(IntWritable.class);
    job3.setOutputValueClass(Text.class);
 
    MultipleInputs.addInputPath(job3, new Path(args[1] + "/step2"), TextInputFormat.class, Mapper31.class);
    MultipleInputs.addInputPath(job3, new Path(args[0]), TextInputFormat.class, Mapper22.class);
    FileOutputFormat.setOutputPath(job3, new Path(args[1] + "/step3"));
    job3.waitForCompletion(true);
    return 0;
  }
  public static void main(String[] args) throws Exception {
    Configuration conf = new Configuration();
    int res = ToolRunner.run(conf, new Lesson1(), args);
    System.exit(res);
  }
}
