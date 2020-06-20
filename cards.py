import random
import collections

def get_suits():
    return ['C','D','H','S'] #Clubs, Diamonds, Hearts, Spades

def get_values():
    return ['A','2','3','4','5','6','7','8','9','10','J','Q','K']

def create_deck():
    '''Returns a list representing a standard deck of playing cards'''
    suits = get_suits()
    values = get_values()
    deck = []
    for suit in suits:
        for value in values:
            deck.append(suit + '-' + value)
    return deck

def deal_hand(deck):
    '''Returns a hand of five playing cards'''
    hand = []
    while len(hand) < 5:
        card = random.choice(deck)
        if card not in hand:
            hand.append(card)
    return hand

def four_of_a_kind(hand):
    '''Return True if four of the five cards have the same value.'''
    values = [card[2:] for card in hand]
    values.sort()
    idx1,idx2=0,0
    maxsamevalcards=1
    while idx1<5 and idx2<5:
        if idx1==2:
            break
        if values[idx1]==values[idx2]:
            if idx2==4:
                maxsamevalcards = max(maxsamevalcards, idx2 - idx1 + 1)
            idx2+=1
        else:
            maxsamevalcards = max(maxsamevalcards,idx2-idx1)
            idx1=idx2

    if maxsamevalcards == 4:
        return True
    return False


def is_flush(hand):
    '''Return True if all five cards are the same suit.'''
    suit_set = set([suit[:1] for suit in hand])
    if len(suit_set) == 1:
        return True
    return False


def get_int_value(card):
    '''Return integer value of card.'''
    card_value = card[2:]
    values = get_values()
    for i, value in enumerate(values, 1):
        if value == card_value:
            return i

def has_ace(hand):
    for card in hand:
        if card[2:] == 'A':
            return True
    return False

def high_card(hand):
    '''Return highest value card in hand.'''
    values = get_values()
    if has_ace(hand):
        return 'A'
    else:
        i = max([get_int_value(card) for card in hand])-1 #replace 0 with correct statement
        return values[i]


def main():
    deck = create_deck()
    hand = deal_hand(deck)

    print(hand,high_card(hand))
    if is_flush(hand):
        print('Congrats! A flush!')
    elif four_of_a_kind(hand):
        print('Congrats! Four of a kind!')


if __name__ == '__main__':
    main()