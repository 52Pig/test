package com.nowcoder.course;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
public class Reducer2 extends Reducer<IntWritable, Text, IntWritable, Text>{
  public void reduce(IntWritable userId, Iterable<Text> values, Context context) throws IOException,
  InterruptedException {
    int sum = 0;
    List<String> neighbors = new ArrayList<String>();
    for (Text val : values) {
      String valStr = val.toString();
      if(valStr.contains(",")){
        neighbors.add(valStr);
      } else {
        sum += Integer.parseInt(valStr);
      }
    }
    StringBuffer result = new StringBuffer();
    for (String neighbor : neighbors) {
      if(result.length() > 0){
        result.append(";");
      }
      result.append(neighbor);
      result.append(",");
      result.append(sum);
    }
    context.write(userId, new Text(result.toString()));
  }
}
