import matplotlib.pyplot as plt
from utils import evaluate_classify
import numpy as np

zero = [0.04602374508976936, 0.045047976076602936, 0.043584324419498444, 0.04651162773370743, 0.04586111381649971, 0.04228329658508301, 0.036916572600603104, 0.032037731260061264, 0.028947796672582626, 0.030248820781707764, 0.034802407026290894, 0.04683688282966614, 0.07058057934045792, 0.12701252102851868, 0.21092860400676727, 0.3407058119773865, 0.5132541656494141, 0.6563668847084045, 0.7025532722473145, 0.6407545804977417, 0.4532444179058075, 0.2693120837211609, 0.1771019697189331, 0.12408521771430969, 0.09773947298526764, 0.07513416558504105, 0.06017238646745682, 0.04976418986916542, 0.043584324419498444, 0.03984387591481209, 0.04212066903710365, 0.04521060362458229, 0.045373231172561646, 0.04456008970737457, 0.042933810502290726, 0.042933810502290726, 0.04602374508976936]
one = [0.1333550214767456, 0.15156936645507812, 0.19921939074993134, 0.302325576543808, 0.47487396001815796, 0.644169807434082, 0.6910066604614258, 0.6179866790771484, 0.44120994210243225, 0.27516669034957886, 0.18198080360889435, 0.14181168377399445, 0.1333550214767456, 0.15156936645507812, 0.19921939074993134, 0.302325576543808, 0.47487396001815796, 0.644169807434082, 0.6910066604614258, 0.6179866790771484, 0.44120994210243225, 0.27516669034957886, 0.18198080360889435, 0.14181168377399445, 0.1333550214767456, 0.15156936645507812, 0.19921939074993134, 0.302325576543808, 0.47487396001815796, 0.644169807434082, 0.6910066604614258, 0.6179866790771484, 0.44120994210243225, 0.27516669034957886, 0.18198080360889435, 0.14181168377399445, 0.1333550214767456]
two = [0.12294682115316391, 0.13904699683189392, 0.20442348718643188, 0.31143274903297424, 0.48495689034461975, 0.655879020690918, 0.7064563632011414, 0.6363636255264282, 0.44657668471336365, 0.276792973279953, 0.1728736311197281, 0.13579443097114563, 0.12294682115316391, 0.13904699683189392, 0.20442348718643188, 0.31143274903297424, 0.48495689034461975, 0.655879020690918, 0.7064563632011414, 0.6363636255264282, 0.4467393159866333, 0.276792973279953, 0.1728736311197281, 0.13563181459903717, 0.12294682115316391, 0.13904699683189392, 0.20442348718643188, 0.31143274903297424, 0.48495689034461975, 0.655879020690918, 0.7064563632011414, 0.6363636255264282, 0.4467393159866333, 0.27663034200668335, 0.1728736311197281, 0.13563181459903717, 0.12294682115316391]
three = [0.10538298636674881, 0.12408521771430969, 0.18116766214370728, 0.2928931415081024, 0.48885998129844666, 0.6539274454116821, 0.7092210054397583, 0.6431940197944641, 0.4683688282966614, 0.29842251539230347, 0.18263132870197296, 0.12278418987989426, 0.10538298636674881, 0.12408521771430969, 0.18116766214370728, 0.2928931415081024, 0.48885998129844666, 0.6537648439407349, 0.7092210054397583, 0.6431940197944641, 0.4683688282966614, 0.2982598841190338, 0.18263132870197296, 0.12278418987989426, 0.10538298636674881, 0.12408521771430969, 0.18116766214370728, 0.2928931415081024, 0.48885998129844666, 0.6537648439407349, 0.7092210054397583, 0.6431940197944641, 0.4683688282966614, 0.2982598841190338, 0.18263132870197296, 0.12278418987989426, 0.10538298636674881]
four = [0.10765977948904037, 0.11709220707416534, 0.17645145952701569, 0.30818018317222595, 0.491136759519577, 0.6527890563011169, 0.70108962059021, 0.6319726705551147, 0.45373231172561646, 0.27402830123901367, 0.17726460099220276, 0.1279882937669754, 0.10765977948904037, 0.11709220707416534, 0.17645145952701569, 0.30818018317222595, 0.491136759519577, 0.6527890563011169, 0.70108962059021, 0.6319726705551147, 0.45373231172561646, 0.2741909325122833, 0.17726460099220276, 0.1279882937669754, 0.10765977948904037, 0.11709220707416534, 0.17645145952701569, 0.30818018317222595, 0.491136759519577, 0.6527890563011169, 0.70108962059021, 0.6319726705551147, 0.45373231172561646, 0.27402830123901367, 0.17726460099220276, 0.1279882937669754, 0.10765977948904037]

zero_jitter = [0.7154008746147156, 0.714913010597229, 0.7142624855041504, 0.7163766622543335, 0.7180029153823853, 0.7186534404754639, 0.7199544906616211, 0.720767617225647, 0.7241827845573425, 0.7212554812431335, 0.7219060063362122, 0.7210928797721863, 0.7160513997077942, 0.7168645262718201, 0.714913010597229, 0.7142624855041504, 0.7201170921325684, 0.7217433452606201, 0.7248333096504211, 0.7243454456329346, 0.7232069969177246, 0.7219060063362122, 0.7222312688827515, 0.7209302186965942, 0.7175150513648987, 0.7158887386322021, 0.7152382731437683, 0.7163766622543335, 0.717677652835846, 0.7189787030220032, 0.7225565314292908, 0.7204423546791077, 0.7209302186965942, 0.7202796936035156, 0.7204423546791077, 0.7163766622543335, 0.7154008746147156]
one_jitter = [0.7240201830863953, 0.7269474864006042, 0.7258090972900391, 0.7261343598365784, 0.7272727489471436, 0.7266222238540649, 0.7267848253250122, 0.7284111380577087, 0.7302000522613525, 0.7253211736679077, 0.7240201830863953, 0.7246706485748291, 0.7240201830863953, 0.7269474864006042, 0.7258090972900391, 0.7261343598365784, 0.7272727489471436, 0.7266222238540649, 0.7266222238540649, 0.7284111380577087, 0.7302000522613525, 0.7253211736679077, 0.7240201830863953, 0.7246706485748291, 0.7240201830863953, 0.7269474864006042, 0.7258090972900391, 0.7261343598365784, 0.7272727489471436, 0.7266222238540649, 0.7267848253250122, 0.7284111380577087, 0.7302000522613525, 0.7253211736679077, 0.7240201830863953, 0.7246706485748291, 0.7240201830863953]
two_jitter = [0.7449991703033447, 0.743372917175293, 0.7441860437393188, 0.7436981797218323, 0.7451618313789368, 0.7472760081291199, 0.7476012110710144, 0.7456496953964233, 0.7430476546287537, 0.7440234422683716, 0.7428849935531616, 0.7422345280647278, 0.7449991703033447, 0.743372917175293, 0.7441860437393188, 0.7436981797218323, 0.7451618313789368, 0.7472760081291199, 0.7476012110710144, 0.7456496953964233, 0.7430476546287537, 0.7440234422683716, 0.7428849935531616, 0.7422345280647278, 0.7449991703033447, 0.743372917175293, 0.7441860437393188, 0.7436981797218323, 0.7451618313789368, 0.7472760081291199, 0.7476012110710144, 0.7456496953964233, 0.7430476546287537, 0.7440234422683716, 0.7428849935531616, 0.7422345280647278, 0.7449991703033447]
three_jitter = [0.7445113062858582, 0.7459749579429626, 0.7449991703033447, 0.7446739077568054, 0.7459749579429626, 0.7492275238037109, 0.7510164380073547, 0.7506911754608154, 0.7497153878211975, 0.7495527863502502, 0.7490648627281189, 0.7449991703033447, 0.7445113062858582, 0.7459749579429626, 0.7449991703033447, 0.7446739077568054, 0.7459749579429626, 0.7492275238037109, 0.7510164380073547, 0.7506911754608154, 0.7497153878211975, 0.7495527863502502, 0.7490648627281189, 0.7449991703033447, 0.7445113062858582, 0.7459749579429626, 0.7448365688323975, 0.7446739077568054, 0.7459749579429626, 0.7492275238037109, 0.7510164380073547, 0.7506911754608154, 0.7497153878211975, 0.7495527863502502, 0.7490648627281189, 0.7449991703033447, 0.7445113062858582]
four_jitter = [0.7536184787750244, 0.7528053522109985, 0.751341700553894, 0.7550821304321289, 0.7557326555252075, 0.7606114745140076, 0.7612619996070862, 0.760123610496521, 0.7584972977638245, 0.7575215697288513, 0.75703364610672, 0.7544316053390503, 0.7536184787750244, 0.7528053522109985, 0.751341700553894, 0.7550821304321289, 0.7557326555252075, 0.7609367370605469, 0.7612619996070862, 0.759960949420929, 0.7584972977638245, 0.7575215697288513, 0.75703364610672, 0.754269003868103, 0.7536184787750244, 0.7528053522109985, 0.7515043020248413, 0.7550821304321289, 0.7557326555252075, 0.7607741355895996, 0.7612619996070862, 0.760123610496521, 0.7584972977638245, 0.7575215697288513, 0.75703364610672, 0.7544316053390503, 0.7536184787750244]

zero_stl = [0.6576250195503235, 0.672124981880188, 0.6676250100135803, 0.6647499799728394, 0.6647499799728394, 0.65625, 0.6480000019073486, 0.6387500166893005, 0.6296250224113464, 0.6365000009536743, 0.6556249856948853, 0.6816250085830688, 0.7120000123977661, 0.7419999837875366, 0.7666249871253967, 0.7953749895095825, 0.8218749761581421, 0.8442500233650208, 0.8519999980926514, 0.843500018119812, 0.8176249861717224, 0.7806249856948853, 0.7387499809265137, 0.7014999985694885, 0.668749988079071, 0.6387500166893005, 0.6088749766349792, 0.5902500152587891, 0.5879999995231628, 0.5973749756813049, 0.6025000214576721, 0.6133750081062317, 0.6202499866485596, 0.6284999847412109, 0.6370000243186951, 0.6453750133514404, 0.6576250195503235]
one_stl = [0.7738749980926514, 0.7828750014305115, 0.7991250157356262, 0.8186249732971191, 0.8382499814033508, 0.8532500267028809, 0.8582500219345093, 0.8508750200271606, 0.8292499780654907, 0.8044999837875366, 0.7823749780654907, 0.7730000019073486, 0.7738749980926514, 0.7828750014305115, 0.7992500066757202, 0.8186249732971191, 0.8382499814033508, 0.8532500267028809, 0.8582500219345093, 0.8508750200271606, 0.8291249871253967, 0.8044999837875366, 0.7823749780654907, 0.7728750109672546, 0.7738749980926514, 0.7828750014305115, 0.7992500066757202, 0.8187500238418579, 0.8382499814033508, 0.8532500267028809, 0.8582500219345093, 0.8508750200271606, 0.8291249871253967, 0.8046249747276306, 0.7823749780654907, 0.7728750109672546, 0.7738749980926514]
two_stl = [0.7350000143051147, 0.7480000257492065, 0.7605000138282776, 0.7836250066757202, 0.8181250095367432, 0.843625009059906, 0.8573750257492065, 0.8453750014305115, 0.8113750219345093, 0.7792500257492065, 0.7551249861717224, 0.7337499856948853, 0.7350000143051147, 0.7480000257492065, 0.7605000138282776, 0.7837499976158142, 0.8181250095367432, 0.843625009059906, 0.8573750257492065, 0.8453750014305115, 0.8113750219345093, 0.7792500257492065, 0.7551249861717224, 0.7338749766349792, 0.7350000143051147, 0.7480000257492065, 0.7605000138282776, 0.7837499976158142, 0.8181250095367432, 0.843625009059906, 0.8573750257492065, 0.8453750014305115, 0.8113750219345093, 0.7792500257492065, 0.7551249861717224, 0.7337499856948853, 0.7350000143051147]
three_stl = [0.6974999904632568, 0.7091249823570251, 0.7362499833106995, 0.7641249895095825, 0.8017500042915344, 0.831125020980835, 0.8410000205039978, 0.8242499828338623, 0.7836250066757202, 0.7578750252723694, 0.7246249914169312, 0.7048749923706055, 0.6974999904632568, 0.7091249823570251, 0.7362499833106995, 0.7641249895095825, 0.8016250133514404, 0.831125020980835, 0.8410000205039978, 0.8242499828338623, 0.7836250066757202, 0.7578750252723694, 0.7246249914169312, 0.7048749923706055, 0.6974999904632568, 0.7091249823570251, 0.7362499833106995, 0.7641249895095825, 0.8016250133514404, 0.831125020980835, 0.8410000205039978, 0.8241249918937683, 0.7836250066757202, 0.7578750252723694, 0.7246249914169312, 0.7048749923706055, 0.6974999904632568]
four_stl = [0.7394999861717224, 0.7476249933242798, 0.7673749923706055, 0.7891250252723694, 0.8130000233650208, 0.8381249904632568, 0.8510000109672546, 0.8418750166893005, 0.812375009059906, 0.7822499871253967, 0.7602499723434448, 0.7413750290870667, 0.7394999861717224, 0.7476249933242798, 0.7674999833106995, 0.7892500162124634, 0.8130000233650208, 0.8381249904632568, 0.8510000109672546, 0.8418750166893005, 0.812375009059906, 0.7822499871253967, 0.7602499723434448, 0.7412499785423279, 0.7394999861717224, 0.7476249933242798, 0.7673749923706055, 0.7892500162124634, 0.8130000233650208, 0.8381249904632568, 0.8510000109672546, 0.8418750166893005, 0.812375009059906, 0.7822499871253967, 0.7602499723434448, 0.7413750290870667, 0.7394999861717224]

zero_stl_jitter = [0.8402500152587891, 0.8411250114440918, 0.843999981880188, 0.8447499871253967, 0.8447499871253967, 0.8451250195503235, 0.844124972820282, 0.843375027179718, 0.8421249985694885, 0.8412500023841858, 0.8417500257492065, 0.8423749804496765, 0.8432499766349792, 0.8450000286102295, 0.8464999794960022, 0.8466249704360962, 0.8472499847412109, 0.8462499976158142, 0.8456249833106995, 0.843999981880188, 0.8427500128746033, 0.8416249752044678, 0.8427500128746033, 0.8422499895095825, 0.8427500128746033, 0.8443750143051147, 0.8432499766349792, 0.8454999923706055, 0.8447499871253967, 0.8448749780654907, 0.8448749780654907, 0.8446249961853027, 0.8420000076293945, 0.8398749828338623, 0.8396250009536743, 0.8416249752044678, 0.8402500152587891]
one_stl_jitter = [0.8544999957084656, 0.8535000085830688, 0.8521249890327454, 0.8543750047683716, 0.8556249737739563, 0.8573750257492065, 0.8567500114440918, 0.8577499985694885, 0.8573750257492065, 0.8546249866485596, 0.8553749918937683, 0.8539999723434448, 0.8544999957084656, 0.8535000085830688, 0.8521249890327454, 0.8542500138282776, 0.8554999828338623, 0.8572499752044678, 0.8567500114440918, 0.8576250076293945, 0.8573750257492065, 0.8546249866485596, 0.8554999828338623, 0.8539999723434448, 0.8544999957084656, 0.8535000085830688, 0.8519999980926514, 0.8542500138282776, 0.8556249737739563, 0.8573750257492065, 0.8567500114440918, 0.8577499985694885, 0.8573750257492065, 0.8546249866485596, 0.8553749918937683, 0.8538749814033508, 0.8544999957084656]
two_stl_jitter = [0.8518750071525574, 0.8518750071525574, 0.8519999980926514, 0.8533750176429749, 0.8539999723434448, 0.8546249866485596, 0.8546249866485596, 0.8551250100135803, 0.8553749918937683, 0.8553749918937683, 0.8535000085830688, 0.8526250123977661, 0.8518750071525574, 0.8519999980926514, 0.8518750071525574, 0.8533750176429749, 0.8539999723434448, 0.8546249866485596, 0.8546249866485596, 0.8550000190734863, 0.8553749918937683, 0.8554999828338623, 0.8535000085830688, 0.8526250123977661, 0.8518750071525574, 0.8518750071525574, 0.8519999980926514, 0.8533750176429749, 0.8539999723434448, 0.8546249866485596, 0.8546249866485596, 0.8551250100135803, 0.8553749918937683, 0.8554999828338623, 0.8536249995231628, 0.8526250123977661, 0.8518750071525574]
three_stl_jitter = [0.8476250171661377, 0.8477500081062317, 0.8462499976158142, 0.8482499718666077, 0.8486250042915344, 0.8485000133514404, 0.8482499718666077, 0.8472499847412109, 0.8463749885559082, 0.8463749885559082, 0.8476250171661377, 0.8482499718666077, 0.8476250171661377, 0.8477500081062317, 0.8462499976158142, 0.8482499718666077, 0.8486250042915344, 0.8485000133514404, 0.8482499718666077, 0.8472499847412109, 0.8463749885559082, 0.8463749885559082, 0.8476250171661377, 0.8483750224113464, 0.8476250171661377, 0.8477500081062317, 0.8462499976158142, 0.8481249809265137, 0.8486250042915344, 0.8485000133514404, 0.8482499718666077, 0.8472499847412109, 0.8463749885559082, 0.8463749885559082, 0.8476250171661377, 0.8483750224113464, 0.8476250171661377]
four_stl_jitter = [0.8421249985694885, 0.8421249985694885, 0.8416249752044678, 0.843625009059906, 0.8426250219345093, 0.844124972820282, 0.8446249961853027, 0.8432499766349792, 0.8432499766349792, 0.8424999713897705, 0.8412500023841858, 0.8411250114440918, 0.8421249985694885, 0.8421249985694885, 0.8416249752044678, 0.843625009059906, 0.8426250219345093, 0.844124972820282, 0.8447499871253967, 0.843375027179718, 0.843375027179718, 0.8426250219345093, 0.8411250114440918, 0.8411250114440918, 0.8421249985694885, 0.8421249985694885, 0.8416249752044678, 0.843625009059906, 0.8426250219345093, 0.844124972820282, 0.8446249961853027, 0.843375027179718, 0.843375027179718, 0.8426250219345093, 0.8412500023841858, 0.8411250114440918, 0.8421249985694885]

def preprocess_data(zero_stages, one_stage, two_stages, three_stages, four_stages):
    values = [0] # baseline is 0
    key = int(len(zero_stages) / 2)
    baseline = zero_stages[key]
    values.append((one_stage[key] - baseline) * 100)
    values.append((two_stages[key] - baseline) * 100)
    values.append((three_stages[key] - baseline) * 100)
    values.append((four_stages[key] - baseline) * 100)

    return values

# evaluate_classify(path="/home/hannah/Master AI/Deep Learning 2/fig3/flowers102-resnet18_1-false-jitter_0_0-split_1_0-seed_0.pth.tar.ckpt", ce_stages=None, seperable=True, width=None)

# y_flower = [0, 0.2, 0.4, -0.1, 0.2]
# y_stl = [0, 0, 0, 0, 0]

# y_flower_jitter = [0, 0.2, 0.4, -0.1, 0.2]
# y_stl_jitter = [0, 0, 0, 0, 0]

def color_selective_datasets_plot(y_flower, y_stl, y_flower_jitter, y_stl_jitter):
    x = [0, 1, 2, 3, 4]
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4))
    plt.subplots_adjust(wspace=0.4, hspace=0.8, top=0.8, bottom=0.25001)

    ax1.plot(x, y_flower, '-D', color='#23c34a')
    ax1.plot(x, y_stl, '-s')
    ax1.grid(axis="x")
    ax1.grid(axis="y")
    ax1.set_ylim(-6, 6)
    ax1.set_title('w/o color-jitter augmentation')
    ax1.set_xlabel('Color Equivariance embedded up to stage')
    ax1.set_ylabel(r'Accuracy improvement' + '\n' + r'(equivariant - vanilla)')

    ax2.plot(x, y_flower_jitter, '-D', color='#23c34a')
    ax2.plot(x, y_stl_jitter, '-s')
    ax2.grid(axis="x")
    ax2.grid(axis="y")
    ax2.set_ylim(-6, 6)
    ax2.set_title('w/ color-jitter augmentation')
    ax2.set_xlabel('Color Equivariance embedded up to stage')
    ax2.set_ylabel(r'Accuracy improvement' + '\n' + r'(equivariant - vanilla)')

    label_names = ["Flowers102 (color sel. 0.70)", "STL10 (color sel. 0.38)"] 
    f.legend([y_flower, y_stl], labels=label_names, 
            loc="lower center") 

    f.suptitle("Color Equivariant statges vs accuracy improvement")

    plt.show()

# flowers_jitter = preprocess_data(zero_jitter, one_jitter, two_jitter, three_jitter, four_jitter)
# stl_jitter = preprocess_data(zero_stl_jitter, one_stl_jitter, two_stl_jitter, three_stl_jitter, four_stl_jitter)
# stl = preprocess_data(zero_stl, one_stl, two_stl, three_stl, four_stl)
# flower = preprocess_data(zero, one, two, three, four)
# color_selective_datasets_plot(flower, stl, flowers_jitter, stl_jitter)

def hue_shifts_plot():
    x = [-0.5, -0.4722222222222222, -0.4444444444444444, -0.4166666666666667, -0.3888888888888889, -0.3611111111111111, -0.33333333333333337, -0.3055555555555556, -0.2777777777777778, -0.25, -0.2222222222222222, -0.19444444444444448, -0.16666666666666669, -0.1388888888888889, -0.11111111111111116, -0.08333333333333337, -0.05555555555555558, -0.02777777777777779, 0.0, 0.02777777777777779, 0.05555555555555558, 0.08333333333333326, 0.11111111111111105, 0.13888888888888884, 0.16666666666666663, 0.19444444444444442, 0.2222222222222222, 0.25, 0.2777777777777777, 0.30555555555555547, 0.33333333333333326, 0.36111111111111105, 0.38888888888888884, 0.41666666666666663, 0.4444444444444444, 0.4722222222222222, 0.5]

    y_1 = [0.04602374508976936, 0.045047976076602936, 0.043584324419498444, 0.04651162773370743, 0.04586111381649971, 0.04228329658508301, 0.036916572600603104, 0.032037731260061264, 0.028947796672582626, 0.030248820781707764, 0.034802407026290894, 0.04683688282966614, 0.07058057934045792, 0.12701252102851868, 0.21092860400676727, 0.3407058119773865, 0.5132541656494141, 0.6563668847084045, 0.7025532722473145, 0.6407545804977417, 0.4532444179058075, 0.2693120837211609, 0.1771019697189331, 0.12408521771430969, 0.09773947298526764, 0.07513416558504105, 0.06017238646745682, 0.04976418986916542, 0.043584324419498444, 0.03984387591481209, 0.04212066903710365, 0.04521060362458229, 0.045373231172561646, 0.04456008970737457, 0.042933810502290726, 0.042933810502290726, 0.04602374508976936]
    y_1_nonorm = [0.04098227247595787, 0.040169134736061096, 0.03854285180568695, 0.04179541394114494, 0.043259065598249435, 0.04212066903710365, 0.03935599327087402, 0.033176127821207047, 0.029923565685749054, 0.02976093627512455, 0.03285086899995804, 0.04423483461141586, 0.06879167258739471, 0.12392258644104004, 0.2104407250881195, 0.3381037712097168, 0.5204098224639893, 0.6625467538833618, 0.7028785347938538, 0.6444950103759766, 0.4556838572025299, 0.2693120837211609, 0.17921613156795502, 0.1315661072731018, 0.10099203139543533, 0.07806147634983063, 0.06505122780799866, 0.05285412445664406, 0.047812651842832565, 0.045047976076602936, 0.04423483461141586, 0.04862579330801964, 0.05073995888233185, 0.048137906938791275, 0.04130753129720688, 0.04309643805027008, 0.04098227247595787]

    y_5 = [0.3283460736274719, 0.383964866399765, 0.5194340348243713, 0.655879020690918, 0.6783216595649719, 0.5670840740203857, 0.41827940940856934, 0.34981298446655273, 0.361847460269928, 0.47747600078582764, 0.59895920753479, 0.6179866790771484, 0.5147178173065186, 0.3740445673465729, 0.3309481143951416, 0.3746950626373291, 0.5267522931098938, 0.6779963970184326, 0.7243454456329346, 0.6521385312080383, 0.4750365912914276, 0.3420068323612213, 0.32281672954559326, 0.406570166349411, 0.5300048589706421, 0.5992844104766846, 0.58204585313797, 0.4561717212200165, 0.3631484806537628, 0.36119693517684937, 0.4467393159866333, 0.576353907585144, 0.6713286638259888, 0.6459587216377258, 0.4815417230129242, 0.3545292019844055, 0.3283460736274719]
    y_5_nonorm = [0.3237924873828888, 0.38494065403938293, 0.5124410390853882, 0.6480728387832642, 0.6744186282157898, 0.5524475574493408, 0.4044560194015503, 0.33729061484336853, 0.35339078307151794, 0.47064563632011414, 0.5966823697090149, 0.618637204170227, 0.5062611699104309, 0.3678646981716156, 0.32281672954559326, 0.37843552231788635, 0.5069116950035095, 0.6745812296867371, 0.7230443954467773, 0.6431940197944641, 0.45633435249328613, 0.33322492241859436, 0.31533583998680115, 0.4055944085121155, 0.5295169949531555, 0.6064400672912598, 0.5825337171554565, 0.45194339752197266, 0.35257765650749207, 0.3564807176589966, 0.43909579515457153, 0.5747275948524475, 0.6718165278434753, 0.6483981013298035, 0.4797528088092804, 0.33973002433776855, 0.3237924873828888]

    y_10 = [0.5981460213661194, 0.5615547299385071, 0.573263943195343, 0.6586436629295349, 0.70596843957901, 0.689380407333374, 0.7240201830863953, 0.7355667352676392, 0.6989754438400269, 0.6266059279441833, 0.630183756351471, 0.6105057597160339, 0.5467555522918701, 0.5469182133674622, 0.6035127639770508, 0.6456334590911865, 0.6483981013298035, 0.7173523902893066, 0.746300220489502, 0.714913010597229, 0.6601073145866394, 0.6605952382087708, 0.6358757615089417, 0.564319372177124, 0.5607416033744812, 0.58806312084198, 0.5942429900169373, 0.6083915829658508, 0.695235013961792, 0.734428346157074, 0.7106846570968628, 0.6988128423690796, 0.7046674489974976, 0.6677508354187012, 0.5961945056915283, 0.5978207588195801, 0.5981460213661194]

    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,10))
    plt.subplots_adjust(wspace=0.4, hspace=0.3)
    
    ax1.plot(x, y_1, color='#2469c8', label='CE-1')
    ax1.plot(x, y_5, color='#d52320', label='CE-2')
    ax1.plot(x, y_10, color='#23c34a', label='CE-3')
    ax1.grid(axis="x")
    ax1.grid(axis="y")
    ax1.set_ylim(-0.15, 0.78)
    ax1.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    ax1.set_yticklabels([0, 10, 20, 30, 40, 50, 60, 70])
    ax1.set_xticks([-0.45, -0.3, -0.15, 0.0, 0.15, 0.3, 0.45])
    ax1.set_xticklabels([-150, -100, -50, 0, 50, 100, 150])
    ax1.set_title('Effect of hue rotations with reprojection - Flowers-102')
    ax1.set_xlabel('Test-time hue shift (°)')
    ax1.set_ylabel('Test accuracy (%)')

    ax2.plot(x, y_1_nonorm, color='#2469c8', label='CE-1')
    ax2.plot(x, y_5_nonorm, color='#d52320', label='CE-2')
    ax2.grid(axis="x")
    ax2.grid(axis="y")
    ax2.set_ylim(-0.15, 0.78)
    ax2.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    ax2.set_yticklabels([0, 10, 20, 30, 40, 50, 60, 70])
    ax2.set_xticks([-0.45, -0.3, -0.15, 0.0, 0.15, 0.3, 0.45])
    ax2.set_xticklabels([-150, -100, -50, 0, 50, 100, 150])
    ax2.set_title('Effect of hue rotations without reprojection - Flowers-102')
    ax2.set_xlabel('Test-time hue shift (°)')
    ax2.set_ylabel('Test accuracy (%)')
    
    ax1.legend(loc='lower center', ncol=3)
    ax2.legend(loc='lower center', ncol=3)
 
    plt.show()


hue_shifts_plot()

