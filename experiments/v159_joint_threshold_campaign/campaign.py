"""V159 evidence-first stop implementation. No upload or bypass switch exists."""
import argparse
import copy
import json
import re
from bridge import *
from assessment import (above, score, input_paths, capacity_audit, build_catalog,
                        mathematical_audit, evaluate_snapshot)


def initialize(out=HERE):
    out = Path(out)
    if (out/'campaign.json').exists():
        return Campaign(out).audit()
    require(not (out/'online_scores.json').exists() and not (out/'submission_manifest.json').exists(),
            'Partial campaign already contains a ledger/manifest; refusing to reset attempts')
    c = previous()
    require(c.champ['score'] == '0.930589' and c.remaining == 6,
            'Starting public champion/quota changed; audit before a new campaign')
    require(c.champ['tp'] == 972 and c.champ['predictions'] == 1045, 'Starting counts changed')
    require(len(c.s) == 1 and c.safe_baseline is not None, 'Safe final not confirmed')
    sources = dict(c.cfg['source_hashes'])
    # Freeze V158's real results, code and research; never write to its directory.
    for p in PREVIOUS.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and p.name != 'operation.lock':
            sources[str(p)] = sha(p)
    for p in input_paths():
        require(p.exists(), 'Missing required evidence: '+str(p))
        sources[str(p)] = sha(p)
    catalog = build_catalog(c)
    capacity = capacity_audit()
    cfg = dict(version=159,created_at=now(),baseline=c.champ,
               ground_truth_count=1044,strict_threshold='0.938154',minimum_display_score='0.938155',
               inherited_remaining=6,authorized_parent_total=8,parent_attempts_used=2,
               quota_origin='Inherited V158 six unused attempts, not six newly authorized attempts',
               deadline_at='2026-09-14T15:52:00+08:00',daily_limit=2,
               source_hashes=sources,catalog_sha256=digest(catalog),capacity_sha256=digest(capacity),
               protocol=dict(max_probes=5,reserved_final=1,success_lower95=.8,
                   interval='two-sided Clopper-Pearson 95%',bootstrap_iterations=2000,
                   independent_task_orders=546,stop_on_insufficient_evidence=True,
                   new_model_training=False,route_a='5 main probes + final',
                   route_b='1 auxiliary group + 4 main probes + final'))
    # Configuration is written last; incomplete initialization is not an active campaign.
    write(out/'candidate_catalog.json',catalog)
    write(out/'validation_capacity.json',capacity)
    write(out/'online_scores.json',dict(records=[]))
    write(out/'submission_manifest.json',dict(files=[]))
    write(out/'campaign.json',cfg)
    return Campaign(out).audit()


class Campaign:
    def __init__(self,out=HERE):
        self.out = Path(out)
        self.cfg = read(self.out/'campaign.json')
        verify_sources(self.cfg['source_hashes'])
        self.previous = previous()
        require(self.previous.remaining == self.cfg['inherited_remaining'], 'Parent quota drift')
        self.catalog = read(self.out/'candidate_catalog.json')
        self.capacity = read(self.out/'validation_capacity.json')
        require(digest(self.catalog) == self.cfg['catalog_sha256'], 'Frozen catalog changed')
        require(digest(self.capacity) == self.cfg['capacity_sha256'], 'Frozen capacity evidence changed')
        self.ledger = read(self.out/'online_scores.json')
        self.manifest = read(self.out/'submission_manifest.json')
        self.entries = {e['probe_id']:e for e in self.manifest['files']}
        require(len(self.entries) == len(self.manifest['files']), 'Duplicate prepared artifact')
        self.champ = copy.deepcopy(self.cfg['baseline'])
        observations = list(self.previous.observations)
        seen = set()
        self.blocked = []
        for r in self.ledger['records']:
            require(r['attempt_id'] not in seen, 'Duplicate attempt ID')
            seen.add(r['attempt_id'])
            e = self.entries[r['probe_id']]
            validate_file(e['file'],self.cfg['baseline']['file'],e['actions'],e['sha256'])
            require(r['sha256'] == e['sha256'] and r['quota_cost'] == 1, 'Recorded artifact/quota drift')
            if r['status'] != 'accepted':
                self.blocked.append(r['attempt_id'])
                continue
            tp = infer(r['score'],e['predictions'])
            require(tp == r['tp'], 'Recorded TP drift')
            _,_,nodes = csv_nodes(e['file'])
            coeff = dict.fromkeys((self.previous.eq.index[k] for k in nodes),1)
            observations.append((coeff,tp))
            if score(tp,e['predictions']) > score(self.champ['tp'],self.champ['predictions']):
                self.champ = dict(file=e['file'],sha256=e['sha256'],tp=tp,
                                  predictions=e['predictions'],score=r['score'],
                                  source_class='user_reported_online')
        self.eq = Equations(self.previous.system,observations)
        self.used = len(seen)
        self.remaining = max(0,self.cfg['inherited_remaining']-self.used)

    def expired(self):
        return datetime.now(TZ) >= datetime.fromisoformat(self.cfg['deadline_at'])

    def evaluation(self):
        p = self.out/'evaluation.json'
        if not p.exists():
            return None
        seal = read(self.out/'evaluation_seal.json')
        require(sha(p) == seal['sha256'], 'Evaluation result changed')
        result = read(p)
        require(result['configuration_sha256'] == digest(self.cfg), 'Evaluation configuration mismatch')
        require(not result['passed'] and not result['release_authorized'],
                'This frozen evidence-first assessment has no valid release certificate')
        return result

    def audit(self):
        evaluation = self.evaluation()
        daily_rows = self.previous.cfg['historical_attempts']+self.previous.ledger['records']+self.ledger['records']
        today = datetime.now(TZ).date()
        daily = sum(datetime.fromisoformat(r['submitted_at']).astimezone(TZ).date() == today
                    for r in daily_rows if r.get('submitted_at') and not r.get('duplicate_of'))
        reached = above(self.champ['tp'],self.champ['predictions'],self.cfg['strict_threshold']) and \
                  self.champ['score'] >= self.cfg['minimum_display_score']
        decision = 'await_evidence_evaluation'
        if evaluation:
            decision = 'stop_insufficient_validation_evidence'
        if self.expired() and not evaluation:
            decision = 'stop_original_deadline_expired'
        if not self.remaining:
            decision = 'stop_budget'
        if reached:
            decision = 'stop_target_achieved'
        if self.blocked:
            decision = 'pause_failed_or_anomalous_attempt'
        return dict(actual_champion=self.champ,strict_threshold=self.cfg['strict_threshold'],
                    minimum_display_score=self.cfg['minimum_display_score'],target_reached=reached,
                    inherited_remaining=self.cfg['inherited_remaining'],new_attempts_used=self.used,
                    remaining_submissions=self.remaining,total_parent_attempts_used=2+self.used,
                    reserved_final=1,daily_used_report_time_estimate=daily,
                    platform_quota_live_checked=False,daily_limit=self.cfg['daily_limit'],
                    deadline_at=self.cfg['deadline_at'],deadline_expired=self.expired(),
                    decision=decision,evidence_gate_passed=False,prepared_files=len(self.entries),
                    safe_feasible_worlds=len(self.previous.s),main_total_true=17,
                    main_candidates=len(self.catalog['main']),auxiliary_candidates=len(self.catalog['auxiliary']),
                    blocked_attempts=self.blocked,source_hashes_verified=len(self.cfg['source_hashes']))

    def evaluate(self):
        cached = self.evaluation()
        if cached:
            return cached
        # Even after deadline, finishing the read-only evidence audit is safe. Search is disabled.
        mathematical = mathematical_audit(self.previous,self.catalog)
        result = evaluate_snapshot(self.capacity,mathematical,self.expired())
        result.update(at=now(),configuration_sha256=digest(self.cfg))
        write(self.out/'evaluation.json',result)
        write(self.out/'evaluation_seal.json',dict(sha256=sha(self.out/'evaluation.json')))
        write(self.out/'recommendation.json',self.audit())
        self.report(result)
        return result

    def report(self,result):
        cap = result['capacity']; math = result['mathematics']
        lines = ['# V159 执行结果：证据不足，保留六次', '',
                 '当前最高用户实报 **0.930589，P=1045、TP=972**。目标为线上六位成绩 ≥0.938155。', '',
                 '**决定：不提交。** 验证证据量未通过用户规定的门槛，未生成正式 CSV，未运行平台上传。', '',
                 '## 已完成的验证', '',
                 f'- 冻结并验证 {len(self.cfg["source_hashes"])} 个历史/研究输入文件；历史账本未修改。',
                 '- 复核 0.929119 → 唯一状态 → 0.930589，继承剩余六次，不增加额度。',
                 '- V150 排除已证伪的原始 ID 1、27 后剩 75 项，完整整数约束确定其中恰好 17 真。',
                 '- 辅助池 3 删除、2 替换，与 75 项没有工单冲突。',
                 f'- 新增池上限 {math["route_a_oracle"]["f1_upper"]:.6f}；辅助池上限 {math["auxiliary_oracle"]["f1_upper"]:.6f}；联合上限 {math["joint_oracle"]["f1_upper"]:.6f}。均为理想选择上限，不是策略胜率。',
                 '', '## 为什么停止', '',
                 f'指定原始数据只有 {cap["labeled_orders"]} 个有标签工单。忽略站点隔离后，最多能容纳 {cap["order_only_disjoint_task_upper"]} 个互不重叠的 546 工单任务；这只是容量上界，不是已完成的独立验证。', '',
                 f'双侧 95% Clopper–Pearson 区间下界 ≥80%，在全部成功时也至少需要 {cap["minimum_independent_all_success_trials"]} 个独立任务。2/2 全成功的下界仅 {cap["best_case_interval_at_capacity"][0]:.6%}；即使把五折当成五个成功任务，下界也只有 {cap["five_perfect_trials_counterfactual"][0]:.6%}。', '',
                 '真实完整、可比的留出任务结果为 0；没有可报告的实测达标率。五折缓存、同池拆分、重复抽样不能扩充独立样本数。也不能把局部 OOF 增益加到线上冠军当作一次达标。', '',
                 '## 两条路线的状态', '',
                 '| 路线 | 数学审计 | 完整策略重放 | 达标率/区间 |',
                 '|---|---|---|---|',
                 '| A：5 次新增池探针＋合并 | 完成 | 证据量门槛失败，未启动 | 不可估计 |',
                 '| B：辅助组＋4 次新增池探针＋合并 | 完成 | 证据量门槛失败，未启动 | 不可估计 |', '',
                 '2,000 次站点 bootstrap 未运行：没有有效的完整策略留出结果可供重采样。未实施或宣称已通过全深度策略搜索。现有两个模型/动作方案的失败证据继续保留。', '',
                 '## 可执行工具与下一步', '',
                 '`audit` 核对来源与余额；`evaluate` 返回已封存的审计；`recommend` 返回停止决定；`prepare` 被门槛阻止，不产生文件；`record` 仅接受登记文件的真实反馈，拒绝未登记文件。', '',
                 '本版本实现的是计划规定的“证据先审、失败即停”分支。没有假造验证证书，也没有隐藏的绕过开关；通过后的策略搜索和提交生成路径因前置条件不成立而未启用。', '',
                 '截止时间仍为北京时间 2026-09-14 15:52。本次新消耗 0 次；当前六次均保留。0.938154 尚未超过。', '',
                 '完整机器可读证据见 evaluation.json、validation_capacity.json、candidate_catalog.json 和 campaign.json。']
        (self.out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

    def recommend(self):
        state = self.audit()
        write(self.out/'recommendation.json',state)
        return state

    def prepare(self):
        state = self.audit()
        return dict(**state,emitted=False,reason='No valid full-policy holdout certificate; preserve attempts')

    def record(self,pid,attempt_id,reported,submitted_at,evidence,failed=False):
        require(pid in self.entries,'No registered V159 submission with this probe ID; no attempt recorded')
        require(attempt_id and evidence,'Attempt ID and evidence required')
        require(datetime.fromisoformat(submitted_at).tzinfo is not None,'Timezone required; explicitly label proxy time')
        previous_row = next((r for r in self.ledger['records'] if r['attempt_id'] == attempt_id),None)
        if previous_row:
            require((previous_row['probe_id'],previous_row['score']) == (pid,reported), 'Changed duplicate attempt')
            return dict(idempotent=True,**self.audit())
        require(not failed or reported is None,'A failed attempt cannot have a score')
        e = self.entries[pid]
        validate_file(e['file'],self.cfg['baseline']['file'],e['actions'],e['sha256'])
        row = dict(probe_id=pid,attempt_id=attempt_id,score=reported,submitted_at=submitted_at,
                   recorded_at=now(),evidence=evidence,sha256=e['sha256'],quota_cost=1,
                   predictions=e['predictions'],status='failed' if failed else 'accepted')
        if not failed:
            try:
                require(isinstance(reported,str) and re.fullmatch(r'0\.\d{6}|1\.000000',reported),
                        'Exact six-decimal score required')
                tp = infer(reported,e['predictions']);row['tp'] = tp
                _,_,nodes = csv_nodes(e['file'])
                coeff = dict.fromkeys((self.eq.index[k] for k in nodes),1)
                require(self.eq.solve(extra=[(coeff,tp)]) is not None,'Score contradicts history')
            except (ValueError,TypeError) as exc:
                row.update(status='anomaly',error=str(exc))
        self.ledger['records'].append(row)
        write(self.out/'online_scores.json',self.ledger)
        return Campaign(self.out).recommend()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=HERE)
    sub = p.add_subparsers(dest='command',required=True)
    for name in ('init','audit','evaluate','recommend','prepare'):
        sub.add_parser(name)
    r = sub.add_parser('record')
    for name in ('probe-id','attempt-id','submitted-at','evidence'):
        r.add_argument('--'+name,required=True)
    r.add_argument('--score');r.add_argument('--failed',action='store_true')
    a = p.parse_args()
    try:
        if a.command == 'audit':
            result = Campaign(a.out).audit()
        else:
            a.out.mkdir(parents=True,exist_ok=True)
            with lock(a.out):
                if a.command == 'init':
                    result = initialize(a.out)
                else:
                    c = Campaign(a.out)
                    if a.command == 'record':
                        result = c.record(a.probe_id,a.attempt_id,a.score,a.submitted_at,a.evidence,a.failed)
                    else:
                        result = getattr(c,a.command)()
        print(json.dumps(result,ensure_ascii=False,indent=2))
    except (ValueError,FileNotFoundError) as exc:
        print(json.dumps(dict(error=str(exc),decision='stop_and_audit'),ensure_ascii=False))
        raise SystemExit(2)


if __name__ == '__main__':
    main()
