"""unit：agent_pipeline __main__ 守卫——import 不触发索引/交互循环。

agent_pipeline 是全顶层代码的脚本, 曾存在"import 即执行"的隐患(会读 data/ 并阻塞在 input())。
__main__ 守卫落地后, import 只定义 main, 交互循环仅在脚本作为主入口时启动。
"""


class TestAgentPipelineEntry:
    def test_import_defines_main_without_running_pipeline(self):
        """import agent_pipeline 只定义 main, 不跑管线(守卫缺失会在无 stdin 环境抛 EOFError)。"""
        import agent_pipeline
        assert callable(agent_pipeline.main)
        assert agent_pipeline.SYSTEM_PROMPT

    def test_import_succeeds_without_reading_data_dir(self, monkeypatch):
        """import 不读 data/ 目录(读目录逻辑已收敛进 main, import 阶段零文件系统副作用)。"""
        import agent_pipeline
        assert "data_dir" not in vars(agent_pipeline) or agent_pipeline.data_dir is None
