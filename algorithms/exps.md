### 优化运行时间
- 修改了图像处理函数：中途不再使用pyplot，统一使用opencv处理

### pipeline时间统计
Total runtime: 44.42s | Model init: 18.82s | Image load: 0.06s | GroundDINO: 5.88s | SAM: 19.53s | Postprocess: 0.07s