# Notes for velocity tasks

The robot's root pos is not always consistent with the robot which might be out side the robot

hence the robot's root will be different which may be caused by the mjcf (not checked)

```
        # Current base positions in world frame and lift them by z_offset.
        base_pos_w = self.robot.data.link_pos_w[:, 1].clone()  # (N, 3)
        base_pos_w[:, 2] += self.cfg.viz.z_offset
```

here we use the indice `1` for simplicity.