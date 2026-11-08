#!/bin/bash
# 一键清场：杀掉所有占麦克风的语音脚本（ask.py / 旧星宝），不留残留。
#
# 为什么要专门有这个：2026-11-08 现场同时堆了 5 个 ask.py，它们会一起念 TTS、
# 一起抢麦克风，用户听到的"星宝"其实是其中一个在回话，而它根本不会动机器人。
#
#   bash stop_voice.sh          # 只清语音脚本
#   bash stop_voice.sh --list   # 只看看有哪些，不动手
#
# 坑（2026-11-08 实测踩到）：`pgrep -f` 匹配的是**整条命令行**，所以
#   bash -c "bash stop_voice.sh; python3 xingbao.py --talk-only"
# 这种包装命令本身也含 "xingbao.py"，会被自己匹配到 -> **把自己杀掉**，
# 后面的命令全部不执行。所以必须排除"自己和自己的祖先"。

LIST_ONLY=0
[ "$1" = "--list" ] && LIST_ONLY=1

# 注意 ask[.]py 这种写法：正则里用字符类，避免匹配到本脚本自己的命令行。
PATTERN='(ask[.]py|xing[b]ao[.]py)'

# 收集自己的祖先 PID 链（含父 shell、它上面的 sshd 之类）
ANCESTORS=" "
p=$PPID
for _ in 1 2 3 4 5 6 7 8; do
    [ -z "$p" ] && break
    [ "$p" -le 1 ] 2>/dev/null && break
    ANCESTORS="$ANCESTORS$p "
    p=$(awk '{print $4}' "/proc/$p/stat" 2>/dev/null)
done

TARGETS=""
for pid in $(pgrep -f "$PATTERN"); do
    [ "$pid" = "$$" ] && continue
    case "$ANCESTORS" in
        *" $pid "*) continue ;;
    esac
    TARGETS="$TARGETS $pid"
done

if [ -z "$TARGETS" ]; then
    echo "没有语音进程在跑。"
    exit 0
fi

echo "找到这些语音进程："
ps -o pid,etime,args -p $(echo $TARGETS | tr ' ' ',') 2>/dev/null

if [ "$LIST_ONLY" = "1" ]; then
    exit 0
fi

for p in $TARGETS; do
    kill -TERM "$p" 2>/dev/null && echo "  TERM $p"
done
sleep 2
for p in $TARGETS; do
    kill -0 "$p" 2>/dev/null && kill -KILL "$p" 2>/dev/null && echo "  KILL $p"
done

echo
echo "清理后："
LEFT=""
for pid in $(pgrep -f "$PATTERN"); do
    [ "$pid" = "$$" ] && continue
    case "$ANCESTORS" in
        *" $pid "*) continue ;;
    esac
    LEFT="$LEFT $pid"
done
if [ -z "$LEFT" ]; then
    echo "  干净了。可以启动星宝了。"
else
    ps -o pid,args -p $(echo $LEFT | tr ' ' ',') 2>/dev/null
fi
