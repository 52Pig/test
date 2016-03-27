package com.nowcoder.course;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import java.io.IOException;

public class Mapper1 extends Mapper<Object, Text, Text, IntWritable>{
  private final static IntWritable one = new IntWritable(1);
  public void map(Object key, Text line, Context context) throws IOException, InterruptedException {
    String[] parts = line.toString().split("\t");
    //String movie = parts[0];
    String[] userIds = parts[1].split(";");
    for(int i = 0; i < userIds.length; i++){
      for(int j = 0; j < userIds.length; j++){
        if(Long.parseLong(userIds[i]) < Long.parseLong(userIds[j])){
          context.write(new Text(userIds[i] + "," + userIds[j]), one);
        }
      }
    }
 }
}
