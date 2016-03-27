package com.nowcoder.course;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import java.io.IOException;
public class Mapper22 extends Mapper<Object, Text, IntWritable, Text> {
  public void map(Object key, Text line, Context context) throws IOException, InterruptedException {
    String[] parts = line.toString().split("\t");
    if(parts.length < 2){
      return;
    }
    String[] userIds = parts[1].split(";");
    for (int i = 0; i < userIds.length; i++) {
      context.write(new IntWritable(Integer.parseInt(userIds[i])), new Text("1"));
    }
  }
}
