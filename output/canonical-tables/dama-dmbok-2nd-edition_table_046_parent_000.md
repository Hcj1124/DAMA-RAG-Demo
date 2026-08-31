| Method | Source System Requirement | Complexity | Fact Load | Dimension Load | Overlap | Deletes |
| --- | --- | --- | --- | --- | --- | --- |
| Time stamped Delta Load | Changes in the source system are stamped with the system date and time. | Low | Fast | Fast | Yes | No |
| Log Table Delta Load | Source system changes are captured and stored in log tables | Medium | Nominal | Nominal | Yes | Yes |
| Database Transaction Log | Database captures changes in the transaction log | High | Nominal | Nominal | No | Yes |
| Message Delta | Source system changes are published as [near] real-time messages | Extreme | Slow | Slow | No | Yes |
| Full Load | No change indicator, tables extracted in full and compared to identify change | Simple | Slow | Nominal | Yes | Yes |
