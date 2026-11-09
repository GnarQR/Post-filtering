import csv, importlib.util, io, json, sys, tempfile, unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def module(name,filename):
    spec=importlib.util.spec_from_file_location(name,ROOT/'src'/filename)
    m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m
    spec.loader.exec_module(m)
    return m

gen=module('e8h_gen','generate_e8_hard_dataset.py')
fin=module('e8h_fin','finalize_e8_hard_review.py')
rep=module('e8h_rep','report_e8_hard_pairs.py')

class E8HardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seeds=gen.load_plan(ROOT/'data/e8_hard_seed_plan.json')
    def test_plan_balance(self):
        self.assertEqual(len(self.seeds),30)
        self.assertEqual(sum(x['split']=='dev' for x in self.seeds),6)
        self.assertEqual(sum(x['difficulty']=='condition_pair' for x in self.seeds),10)
    def test_dry_run(self):
        self.assertEqual(gen.main(['--plan',str(ROOT/'data/e8_hard_seed_plan.json'),'--dry-run','--limit','2']),0)
    def test_flatten_and_finalize(self):
        records=[]
        for s in self.seeds:
            g={k:'테스트용 질문의 답변은 사례에 맞게 충분한 길이를 갖추고 있습니다.' for k in gen.SCHEMA['required']}
            g['question']='이 소를 관리할 때 어떤 선택을 하는 편이 좋을까요?'
            g['pass_answer']='상태를 확인한 뒤 건강하고 안전하게 관리하는 방안을 선택해 주세요.'
            g['violation_answer']='상태가 위험한 것으로 확인되었더라도 관리를 바꾸지 않고 계속 진행하는 것을 추천합니다.'
            records.append({'seed':s,'generated':g,'model':'fake','signature':'test','generated_at':'now','latency_seconds':0})
        rows=gen.flatten(records)
        self.assertEqual(len(rows),60)
        reviews=[{'case_id':r['case_id'],'question':r['question'],'answer':r['answer'],
                  '검수상태':'승인','최종Gold':'','수정질문':'','수정답변':'',
                  '검수메모':'','근거URL':''} for r in rows]
        final,summary=fin.finalize(rows,reviews)
        self.assertEqual(len(final),60)
        self.assertEqual(summary['gold'],{'PASS':30,'VIOLATION':30})
        self.assertEqual(summary['splits'],{'dev':12,'test':48})
        reviews[0]['검수상태']='제외'
        with self.assertRaisesRegex(ValueError,'한쪽만 제외'):
            fin.finalize(rows,reviews)
        reviews[1]['검수상태']='제외'
        final,summary=fin.finalize(rows,reviews)
        self.assertEqual(len(final),58)
        self.assertEqual(len(summary['excluded_ids']),2)
    def test_wrong_review_question(self):
        records=[]
        for seed in self.seeds:
            g={k:'정상적인 점검 절차를 살펴보고 필요한 선택을 진행하는 것이 좋습니다.' for k in gen.SCHEMA['required']}
            g['question']='이 소를 관리하려면 어떤 방식을 이용하는 것이 좋을까요?'
            g['pass_answer']='먼저 상태를 확인하고 필요한 관리를 진행하는 것을 추천합니다.'
            g['violation_answer']='이상이 남아 있어도 상태를 확인하지 않은 채 계속 진행하기를 추천합니다.'
            records.append({'seed':seed,'generated':g,'model':'fake','signature':'test','generated_at':'now','latency_seconds':0})
        rows=gen.flatten(records)
        review=[{'case_id':r['case_id'],'question':r['question'],'answer':r['answer'],
                 '검수상태':'승인','최종Gold':'','수정질문':'','수정답변':'',
                 '검수메모':'','근거URL':''} for r in rows]
        review[0]['question']='원본과 다른 질문'
        with self.assertRaisesRegex(ValueError,'원본 질문'):
            fin.finalize(rows,review)
    def test_pair_reporting(self):
        data=[{'case_id':'A_PASS','scenario_id':'A','split':'test','difficulty':'condition_pair','gold':'PASS'},
              {'case_id':'A_VIOLATION','scenario_id':'A','split':'test','difficulty':'condition_pair','gold':'VIOLATION'}]
        comp=[{'case_id':'A_PASS','pred_jev':'PASS','pred_lite':'REVIEW','pred_flash':'PASS'},
              {'case_id':'A_VIOLATION','pred_jev':'VIOLATION','pred_lite':'VIOLATION','pred_flash':'PASS'}]
        result=rep.report(comp,data,'test')
        self.assertEqual(result['model_stats']['jev']['all']['pair_accuracy'],1.)
        self.assertEqual(result['model_stats']['lite']['all']['pair_accuracy'],0.)
        self.assertEqual(result['model_stats']['flash']['all']['pair_accuracy'],0.)

if __name__=='__main__':unittest.main()
