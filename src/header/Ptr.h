#pragma once


#include "ObjectID.h"



/*************************************************/
/*												 */
/*						 Ptr					 */
/*												 */
/*************************************************/

template <class _T>
class Ptr {
protected:
	static Object_ID_List _list;
	Object_ID* ID;

	void Create();
	void Clear();

public:
	_T* pt;

	Ptr(_T* p);
	Ptr(const Ptr<_T>& P);
	~Ptr();

	Ptr<_T>& operator=(const Ptr<_T>& p);
	_T* operator->();
};


template <class _T>
Object_ID_List Ptr<_T>::_list;

template <class _T>
void Ptr<_T>::Create() {
	ID = _list.Create();
}

template <class _T>
void Ptr<_T>::Clear() {
	if (ID) {
		if (ID->nCpy > 1) ID->nCpy -= 1;
		else {
			delete pt;

			_list.Erase(ID);
			ID = NULL;
			pt = NULL;
		}
	}
}

template <class _T>
Ptr<_T>::Ptr(_T* p) {
	pt = p;
	Create();
}

template <class _T>
Ptr<_T>::Ptr(const Ptr<_T>& p) {
	ID = p.ID;
	pt = p.pt;

	if (ID) ID->nCpy += 1;
}

template <class _T>
Ptr<_T>::~Ptr() {
	Clear();
}

template <class _T>
Ptr<_T>& Ptr<_T>::operator=(const Ptr<_T>& p) {
	if (this == &p) return *this;

	Clear();

	ID = p.ID;
	pt = p.pt;

	if (ID) ID->nCpy += 1;

	return *this;
}

template <class _T>
_T* Ptr<_T>::operator->() {
	return pt;
}