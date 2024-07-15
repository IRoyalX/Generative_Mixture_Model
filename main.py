from nltk.tokenize import TweetTokenizer
import math

Collection_Address = "HAM2-corpus-all.txt"
Document_Address = "input3.txt"
Lambda1, Lambda2 = 0.01, 0.001

def Read(cAdr):
    try:
        dAdr = cAdr.replace(".txt", ".dict.txt")
        with open(dAdr, 'r', encoding='utf-8') as file:
            file, file2 = file.read().split("//@//") 
        return eval(file), eval(file2)
    
    except FileNotFoundError:
        try:
            with open(cAdr, 'r', encoding='utf-8') as file:
                file = ''.join(reversed(file.read()))
            cTokens = TweetTokenizer().tokenize(file)
            cTokens = [cTokens[i] for i in range(len(cTokens) - 1, -1, -1)]
            
            uDict, bDict = dMaker(cTokens)
            
            with open(dAdr, 'w', encoding='utf-8') as file:
                file.write(str(uDict) + "//@//" + str(bDict))
            return uDict, bDict
        
        except FileNotFoundError:
            print("ERROR: Collection file not found")
            exit(1)
        except Exception as e:
            print("ERROR:", e)
            exit(1)
    except Exception as e:
        print("ERROR:", e)
        exit(1)

def dMaker(tokens):
    uDict = [{}, len(tokens)]
    bDict = [{}, len(tokens) - 1]
    #bDict = [{}, (len(tokens) - 1)*len(tokens)/2]

    for term in tokens:
        if term in uDict[0]:
            uDict[0][term] += 1
        else:
            uDict[0][term] = 1
    
    for i in range(1, len(tokens)):
        phrase = tokens[i] + ' ' + tokens[i - 1]
        if phrase in bDict[0]:
            bDict[0][phrase] += 1
        else:
            bDict[0][phrase] = 1

    # for i in range(len(tokens)):
    #     for j in range(i + 1, len(tokens)):
    #         phrase = tokens[j] + ' ' + tokens[i]
    #         if phrase in bDict[0]:
    #             bDict[0][phrase] += 1
    #         else:
    #             bDict[0][phrase] = 1

    return uDict, bDict

def unigram(Doc, Col, Lambda1 = 0.01):
    dDict, dSize = Doc[0], Doc[1]
    cDict, cSize = Col[0], Col[1]
    scores = {}
    for term, freq in dDict.items():
        if term in cDict:
            scores[term] = freq * math.log2(Lambda1 * (freq/dSize) + (1 - Lambda1) * cDict[term]/cSize)
        else:
            scores[term] = freq * math.log2(Lambda1 * (freq/dSize))
    
    return list(dict(sorted(scores.items(), key=lambda item: item[1])).items())

def bigram(cDoc, dDoc, Col, Lambda1 = 0.01, Lambda2 = 0.001):
    dDict, dSize = cDoc[0], cDoc[1]
    tDict, tSize = dDoc[0], dDoc[1]
    cDict, cSize = Col[0], Col[1]
    scores = {}
    for phrase, freq in dDict.items():
        if phrase in cDict:
            scores[phrase] = freq * math.log2(Lambda1 * (Lambda2 * freq / tDict[phrase.split(' ')[0]] + (1 - Lambda2) * (tDict[phrase.split(' ')[1]] / tSize) + (1 - Lambda1) * cDict[phrase] / cSize))
        else:
            scores[phrase] = freq * math.log2(Lambda1 * (Lambda2 * freq / tDict[phrase.split(' ')[0]] + (1 - Lambda2) * (tDict[phrase.split(' ')[1]] / tSize)))
    
    return list(dict(sorted(scores.items(), key=lambda item: item[1])).items())

def output(uScores, bScores):

    print("\nUnigram Model:\n\n"," {:<5} |   {:<20} |   {:<20}".format("NO", "TERM", "SCORE"))
    print('----------------------------------------------------------------------')
    for i in range(15):
        term = uScores[i][0]
        score = uScores[i][1]
        print("  {:<5} |   {:<20} |   {:<20}".format(i + 1, term, score))
        print('----------------------------------------------------------------------')

    print("\n\n\nBigram Model:\n\n"," {:<5} |   {:<20} |   {:<20}".format("NO", "PHRASE", "SCORE"))
    print('----------------------------------------------------------------------')
    for i in range(15):
        phrase = bScores[i][0]
        score = bScores[i][1]
        print("  {:<5} |   {:<20} |   {:<20}".format(i + 1, phrase, score))
        print('----------------------------------------------------------------------')

if __name__ == "__main__":
    cuDict , cbDicT = Read(Collection_Address)
    duDict , dbDict = Read(Document_Address)
    uScores = unigram(duDict, cuDict, Lambda1)
    bScores = bigram(dbDict, duDict, cbDicT, Lambda1, Lambda2)
    output(uScores, bScores)
    

