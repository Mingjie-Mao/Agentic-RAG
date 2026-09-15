"""Editorial source construction, independent of model/retrieval scores."""

GENRES = ["执行手册", "支持工单复盘", "配置检查表", "操作问答", "交接记录"]


def document_sections(item, note, index):
    name, field_a, value_a, field_b, value_b = item
    policy = [
        f"本节只规定星桥软件的{name}，适用时间从 2026 年 9 月 1 日开始。其他公司的同名服务，以及没有明确生效日期的讨论记录，不适用这里的数值。",
        f"{name}：{field_a}为 {value_a}；{field_b}为 {value_b}。两个数值分别约束各自事项，不相加，也不能用其中一个替代另一个。",
        f"记录{field_a}时，应同时写清{name}的对象和计量单位；记录{field_b}时同样保留单位。对工作日、自然日、小时和分钟不得自行换算成另一种计时口径。比例与数量上限也不能互换。",
    ]
    workflow = [
        f"发起{name}前，先确定组织、环境和处理对象，保存原始请求或申请编号。检查当前记录是否已经处理过，再核对{field_a}对应的限制；重复提交不能绕过这项限制。",
        f"执行中分别记录开始、等待、处理结束和确认结果的时间。如果{name}依赖其他团队，写清等待对象和最后一次确认情况，不把发送通知当作对方已完成。",
        f"收尾时核对{field_b}对应的条件，并保存实际结果与原始记录的关联。异常不能只写“失败”或“已解决”，需要列出已确认的现象、做过的操作和仍未解决的问题。",
    ]
    exception = [
        f"超出{name}的规定范围时，提交例外申请，写明业务影响、所需范围、负责人及结束条件。没有确认前仍按本页规则执行；个别申请获批不能修改其他任务的默认设置。",
        f"检查材料时，注意区分实际{name}记录、界面展示和口头说明。若实际记录与本页数值不一致，保留双方来源并交由负责人核对，不能删掉差异后填写“符合”。",
    ]
    closure = [
        f"本页负责人负责{name}的解释与变更登记。修订需要标明新生效日期和受影响对象；本页不提供旧版本的数值，也不代表未列出的特殊地区已经获得例外。",
        f"验收清单：对象是否正确；{field_a}是否满足；{field_b}是否满足；原始编号能否找到；异常是否有人跟进。验收人应查看记录本身，而不是只读结论。",
        "来源：本项目自建虚构业务资料，用于测试检索、证据与访问权限；不代表真实企业制度。",
    ]
    if index == 0:
        return [
            ("适用规则", policy),
            ("执行步骤", workflow),
            ("常见问题", [note]),
            ("例外与验收", exception + closure),
        ]
    if index == 1:
        return [
            ("工单现象", [note]),
            ("处理记录", workflow),
            ("核对依据", policy),
            ("结单条件", exception + closure),
        ]
    if index == 2:
        return [
            ("检查目的", [note]),
            ("参数与单位", policy),
            ("逐项检查", workflow + exception),
            ("复核记录", closure),
        ]
    if index == 3:
        return [
            ("为什么需要单独核对", [note]),
            (f"{name}按什么规则执行", policy),
            ("实际操作怎么记录", workflow),
            ("遇到例外怎么办", exception + closure),
        ]
    return [
        ("交接背景", [note]),
        ("当前执行口径", policy),
        ("下一班需要检查", workflow),
        ("未完成事项和确认", exception + closure),
    ]
