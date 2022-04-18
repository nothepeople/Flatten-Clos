/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008 INRIA
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */
#include "longer-path-tag.h"

namespace ns3 {

NS_OBJECT_ENSURE_REGISTERED (LongerPathTag);

TypeId 
LongerPathTag::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::LongerPathTag")
    .SetParent<Tag> ()
    .AddConstructor<LongerPathTag> ()
  ;
  return tid;
}

TypeId 
LongerPathTag::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

uint32_t 
LongerPathTag::GetSerializedSize (void) const
{
  return 4;
}

void 
LongerPathTag::Serialize (TagBuffer buf) const
{
  buf.WriteU32 (m_longer);
}

void 
LongerPathTag::Deserialize (TagBuffer buf)
{
  m_longer = buf.ReadU32 ();
}

void 
LongerPathTag::Print (std::ostream &os) const
{
  os << "Longer=" << m_longer;
}

LongerPathTag::LongerPathTag ()
  : Tag () 
{
}

LongerPathTag::LongerPathTag (uint32_t longer)
  : Tag (),
    m_longer (longer)
{
}

void
LongerPathTag::SetLonger (uint32_t longer)
{
  m_longer = longer;
}

uint32_t
LongerPathTag::GetLonger (void) const
{
  return m_longer;
}


} // namespace ns3

