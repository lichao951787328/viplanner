<!--
 * @Author: lichao951787328 951787328@qq.com
 * @Date: 2026-02-11 11:52:28
 * @LastEditors: lichao951787328 951787328@qq.com
 * @LastEditTime: 2026-02-11 11:52:36
 * @FilePath: /viplanner/todo_lists.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
方案 C：使用 Dijkstra 引导的 Cost Map (Geodesic Map)
目前的 MapProcessor 生成的是基于欧氏距离的 Cost Map。可以将其改为基于**测地线距离（Geodesic Distance）**的 Map。
原理： 欧氏距离只看物理距离，测地线距离看“绕过障碍物的距离”。
做法：
不要把原始的 Cost Map 输入网络，而是计算一张 "Cost-to-Go Map"。这张图上每个像素的值，是该点到 Goal 点的最短路径距离（考虑障碍物）。
这样，地图的梯度自然就会指向绕过障碍物的方向，顺着梯度下降就能找到出口。
代价： 训练时数据预处理变慢（对每个样本都要跑一次 Dijkstra/Fast Marching Method），但在 Stage 3 这种难样本上非常有效。
