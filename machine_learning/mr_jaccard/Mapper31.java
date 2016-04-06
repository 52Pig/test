package com.nowcoder.course;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import java.io.IOException;
public class Mapper31 extends Mapper<Object, Text, IntWritable, Text> {
  public void map(Object key, Text line, Context context) throws IOException, InterruptedException {
    String[] parts = line.toString().split("\t");
    if (parts.length < 2) {
      return;
    }
    String[] userInfos = parts[1].split(";");
    for (String userInfo : userInfos) {
      String[] infos = userInfo.split(",");
      context.write(new IntWritable(Integer.parseInt(infos[0])), new Text(parts[0] + ","
          + infos[1] + "," + infos[2]));
    }
  }
}
