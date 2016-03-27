package com.nowcoder.course;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;

public class Mapper21 extends Mapper<Object, Text, IntWritable, Text> {

  public void map(Object key, Text line, Context context) throws IOException, InterruptedException {
    String[] parts = line.toString().split("\t");
    if (parts.length < 2) {
      return;
    }
    String[] userIds = parts[0].split(",");
    if (userIds.length < 2) {
      return;
    }
    context.write(new IntWritable(Integer.parseInt(userIds[0])), new Text(userIds[1] + ","
        + parts[1]));
  }
}
