package com.nowcoder.course;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
public class Reducer3 extends Reducer<IntWritable, Text, Text, Text> {
  public void reduce(IntWritable userId, Iterable<Text> values, Context context)
      throws IOException, InterruptedException {
    int sum = 0;
    List<String> neighbors = new ArrayList<String>();
    for (Text val : values) {
      String valStr = val.toString();
      if (valStr.contains(",")) {
        neighbors.add(valStr);
      } else {
        sum += Integer.parseInt(valStr);
      }
    }
    for (String neighbor : neighbors) {
      String[] parts = neighbor.split(",");
      if (parts.length < 3) {
        continue;
      }
      //jaccard
      double corr =
          Double.parseDouble(parts[1])
              / (Double.parseDouble(parts[2]) + sum - Double.parseDouble(parts[1]));
      context.write(new Text(parts[0] + "," + userId), new Text(corr + ""));
    }
  }
}
